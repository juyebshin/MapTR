from tkinter.messagebox import NO
from turtle import forward
from copy import deepcopy
import torch
from torch import Tensor, nn
import torch.nn.functional as F
import numpy as np

from mmcv.runner.base_module import BaseModule, ModuleList, Sequential
from mmcv.cnn.bricks.registry import TRANSFORMER_LAYER_SEQUENCE, FEEDFORWARD_NETWORK
from mmcv.cnn import build_norm_layer


def MLP(channels: list, do_bn=True, norm_type='BN1d'):
    """ MLP """
    n = len(channels)
    layers = []
    for i in range(1, n):
        layers.append(
            nn.Conv1d(channels[i - 1], channels[i], kernel_size=1, bias=True)
        )
        if i < (n-1):
            if do_bn:
                layers.append(build_norm_layer(dict(type=norm_type), num_features=channels[i])[1])
            layers.append(nn.ReLU())
    
    return Sequential(*layers)

def simple_nms(scores, nms_radius: int):
    """ Fast Non-maximum suppression to remove nearby points """
    assert(nms_radius >= 0)

    def max_pool(x):
        return torch.nn.functional.max_pool2d(
            x, kernel_size=nms_radius*2+1, stride=1, padding=nms_radius)

    zeros = torch.zeros_like(scores)
    max_mask = scores == max_pool(scores)
    for _ in range(2):
        supp_mask = max_pool(max_mask.float()) > 0
        supp_scores = torch.where(supp_mask, zeros, scores)
        new_max_mask = supp_scores == max_pool(supp_scores)
        max_mask = max_mask | (new_max_mask & (~supp_mask))
    return torch.where(max_mask, scores, zeros)

def remove_borders(vertices, scores, border: int, height: int, width: int):
        """ Removes vertices too close to the border """
        mask_h = (vertices[:, 0] >= border) & (vertices[:, 0] < (height - border))
        mask_w = (vertices[:, 1] >= border) & (vertices[:, 1] < (width - border))
        mask = mask_h & mask_w
        return vertices[mask], scores[mask]

def sample_dt(vertices, distance: Tensor, s: int = 8):
    """ Extract distance transform patches around vertices """
    # vertices: # tuple of length (b c), [N, 2(row, col)] tensor, in (25, 50) cell
    # distance: (b c) 200 400 tensor
    # embedding, _ = distance.max(1, keepdim=False) # b, 200, 400
    embedding = distance # 0 ~ 10 -> 0 ~ 1 normalize
    bc, h, w = embedding.shape # (b 3) 200 400
    hc, wc = int(h/s), int(w/s) # 25, 50
    embedding = embedding.reshape(bc, hc, s, wc, s).permute(0, 1, 3, 2, 4) # (b c) 25 8 50 8 -> (b c) 25 50 8 8
    embedding = embedding.reshape(bc, hc, wc, s*s) # (b c) 25 50 64
    # embedding = embedding.reshape(b, hc, wc, -1) # b, 25, 50, 192
    embedding = [e[tuple(vc.t())] for e, vc in zip(embedding, vertices)] # tuple of length (b c), [N, 64] tensor
    return embedding

def sample_feat(vertices, feature: Tensor):
    """ Extract feature patches around vertices """
    # vertices: # tuple of length (b c), [N, 2(row, col)] tensor, in (25, 50) cell
    # feature: (b c) 256 25 50 tensor
    bc, c, h, w = feature.shape # (b c) 256 25 50
    embedding = feature.permute(0, 2, 3, 1) # (b c) 25 50 256
    embedding = [e[tuple(vc.t())] for e, vc in zip(embedding, vertices)] # tuple of length (b c), [N, 256] tensor
    return embedding

def normalize_vertices(vertices: Tensor, image_shape):
    """ Normalize vertices locations in BEV space """
    # vertices: [N, 2] tensor in (x, y): (0~49, 0~24)
    _, height, width = image_shape # b c 25 50
    one = vertices.new_tensor(1) # [1], values 1
    size = torch.stack([one*width, one*height])[None] # [1, 2], values [50, 25]
    center = size / 2.0 # [1, 2], values [25, 12.5]
    return (vertices - center + 0.5) / size # [N, 2] values [-0.5, 0.4975] or [-0.49875, 0.49875]

def top_k_vertices(vertices: Tensor, scores: Tensor, embeddings: Tensor, k: int):
    """
    Returns top-K vertices.

    vertices: [N, 2] tensor (N vertices in xy)
    scores: [N] tensor (N vertex scores)
    embeddings: [N, 64] tensor
    """
    # k: 400
    n_vertices = len(vertices) # N
    embedding_dim = embeddings.shape[1]
    if k >= n_vertices:
        pad_size = k - n_vertices # k - N
        pad_v = torch.ones([pad_size, 2], device=vertices.device, requires_grad=False)
        pad_s = torch.ones([pad_size], device=scores.device, requires_grad=False)
        pad_dt = torch.ones([pad_size, embedding_dim], device=embeddings.device, requires_grad=False)
        vertices, scores, embeddings = torch.cat([vertices, pad_v], dim=0), torch.cat([scores, pad_s], dim=0), torch.cat([embeddings, pad_dt], dim=0)
        mask = torch.zeros([k], dtype=torch.uint8, device=vertices.device)
        mask[:n_vertices] = 1
        return vertices, scores, embeddings, mask # [K, 2], [K], [K]
    scores, indices = torch.topk(scores, k, dim=0)
    mask = torch.ones([k], dtype=torch.uint8, device=vertices.device) # [K]
    return vertices[indices], scores, embeddings[indices], mask # [K, 2], [K], [K]

def attention(query, key, value, mask=None):
    # q, k, v: [b, 64, 4, N], mask: [b, 1, N]
    dim = query.shape[1] # 64
    scores = torch.einsum('bdhn,bdhm->bhnm', query, key) / dim**.5 # [b, 4, N, N], dim**.5 == 8
    if mask is not None:
        mask = torch.einsum('bdn,bdm->bdnm', mask, mask) # [b, 1, N, N]
        scores = scores.masked_fill(mask == 0, -1e9)
    prob = torch.nn.functional.softmax(scores, dim=-1) # [b, 4, N, N]
    return torch.einsum('bhnm,bdhm->bdhn', prob, value), prob # final message passing [b, 64, 4, N]


def log_sinkhorn_iterations(Z, log_mu, log_nu, iters: int):    
    """ Perform Sinkhorn Normalization in Log-space for stability"""
    u, v = torch.zeros_like(log_mu), torch.zeros_like(log_nu)
    for _ in range(iters):
        u = log_mu - torch.logsumexp(Z + v.unsqueeze(1), dim=2)
        v = log_nu - torch.logsumexp(Z + u.unsqueeze(2), dim=1)
    return Z + u.unsqueeze(2) + v.unsqueeze(1)

def log_optimal_transport(scores, alpha, iters: int):
    """ Perform Differentiable Optimal Transport in Log-space for stability"""
    b, m, n = scores.shape # b, N, N
    # m_valid = n_valid = torch.count_nonzero(masks, 1).squeeze(-1) # [b] number of valid
    one = scores.new_tensor(1)
    ms, ns = (m*one).to(scores), (n*one).to(scores) # [b] same as m_valid, n_valid

    bins0 = alpha.expand(b, m, 1) # [b, N, 1]
    bins1 = alpha.expand(b, 1, n) # [b, 1, N]
    alpha = alpha.expand(b, 1, 1) # [b, 1, 1]

    couplings = torch.cat( # [b, N+1, N+1]
        [
            torch.cat([scores, bins0], -1), # [b, N, N+1]
            torch.cat([bins1, alpha], -1)   # [b, 1, N+1]
        ], 1)
    # masks_bins = torch.cat([masks, masks.new_tensor(1).expand(b, 1, 1)], 1) # [b, N+1, 1]

    norm = - (ms + ns).log() # [b]
    log_mu = torch.cat([norm.expand(m), ns.log()[None] + norm]) # [N+1]
    log_nu = torch.cat([norm.expand(n), ms.log()[None] + norm]) # [N+1]
    log_mu, log_nu = log_mu[None].expand(b, -1), log_nu[None].expand(b, -1) # [b, N+1]

    Z = log_sinkhorn_iterations(couplings, log_mu, log_nu, iters)
    Z = Z - norm  # multiply probabilities by M+N
    return Z

# Positional embedding from NeRF: https://github.com/bmild/nerf/blob/18b8aebda6700ed659cb27a0c348b737a5f6ab60/run_nerf_helpers.py
def get_embedder(multires, i=0):

    if i == -1:
        return torch.nn.Identity(), 2

    embed_kwargs = {
        'include_input': True,
        'input_dims': 2,
        'max_freq_log2': multires-1,
        'num_freqs': multires,
        'log_sampling': True,
        'periodic_fns': [torch.sin, torch.cos],
    }

    embedder_obj = Embedder(**embed_kwargs)
    def embed(x, eo=embedder_obj): return eo.embed(x)
    return embed, embedder_obj.out_dim

class Embedder:

    def __init__(self, **kwargs):

        self.kwargs = kwargs
        self.create_embedding_fn()

    def create_embedding_fn(self):

        embed_fns = []
        d = self.kwargs['input_dims']
        out_dim = 0
        if self.kwargs['include_input']:
            embed_fns.append(lambda x: x)
            out_dim += d

        max_freq = self.kwargs['max_freq_log2']
        N_freqs = self.kwargs['num_freqs']

        if self.kwargs['log_sampling']:
            freq_bands = 2.**torch.linspace(0., max_freq, N_freqs)
        else:
            freq_bands = torch.linspace(2.**0., 2.**max_freq, N_freqs)

        for freq in freq_bands:
            for p_fn in self.kwargs['periodic_fns']:
                embed_fns.append(lambda x, p_fn=p_fn,
                                 freq=freq: p_fn(x * freq))
                out_dim += d

        self.embed_fns = embed_fns
        self.out_dim = out_dim

    def embed(self, inputs):
        return torch.cat([fn(inputs) for fn in self.embed_fns], -1)

class MultiHeadedAttention(BaseModule):
    """ Multi-head attention to increase model expressivitiy """
    def __init__(self, num_heads: int, d_model: int):
        super().__init__()
        assert d_model % num_heads == 0
        self.dim = d_model // num_heads # 64
        self.num_heads = num_heads # 4
        self.merge = nn.Conv1d(d_model, d_model, kernel_size=1)
        self.proj = ModuleList([deepcopy(self.merge) for _ in range(3)])

    def forward(self, query, key, value, mask=None):
        # q, k, v: [b, 256, N], mask: [b, 1, N]
        batch_dim = query.size(0) # b
        num_vertices = query.size(2) # N
        if mask is None:
            mask = torch.ones([batch_dim, 1, num_vertices], device=query.device) # [b, 1, N]
        query, key, value = [l(x).view(batch_dim, self.dim, self.num_heads, -1)
                             for l, x in zip(self.proj, (query, key, value))]
        # q, k, v: [b, dim, head, N]
        x, _ = attention(query, key, value, mask) # [b, 64, head, N], [b, head, N, N]
        return self.merge(x.contiguous().view(batch_dim, self.dim*self.num_heads, -1))

class AttentionalPropagation(BaseModule):
    def __init__(self, embed_dim: int, num_heads: int, norm_type='BN1d'):
        super().__init__()
        self.attn = MultiHeadedAttention(num_heads, embed_dim)
        self.mlp = MLP([embed_dim*2, embed_dim*2, embed_dim], norm_type=norm_type)
        nn.init.constant_(self.mlp[-1].bias, 0.0)

    def forward(self, x, source, mask=None):
        # x, source: [b, 256(embed_dim), N]
        # attn(q, k, v)
        message = self.attn(x, source, source, mask) # [b, 256, N], [b, 4, N, N]
        return self.mlp(torch.cat([x, message], dim=1)) # [4, 512, 300] -> [4, 256, 300], [b, 4, N, N]

@TRANSFORMER_LAYER_SEQUENCE.register_module()
class AttentionalGNN(BaseModule):
    def __init__(self, embed_dim: int, layer_names: list, norm_type='BN1d'):
        super().__init__()
        # norm_layer = build_norm_layer(norm_cfg)[1]
        self.layers = ModuleList([
            AttentionalPropagation(embed_dim, 4, norm_type)
            for _ in range(len(layer_names))])
        self.names = layer_names

    def forward(self, embedding, mask=None):
        # Only self-attention is implemented for now
        # embedding: [b, 256, N]
        # mask: [b, 1, N]
        for layer, name in zip(self.layers, self.names):
            delta = layer(embedding, embedding, mask) # [b, 256, N], [b, 4, N, N]
            embedding = (embedding + delta) # [b, 256, N]
        return embedding

@FEEDFORWARD_NETWORK.register_module()
class GraphEncoder(BaseModule):
    """ Joint encoding of vertices and distance transform embeddings """
    def __init__(self, embed_dim, layers: list, norm_type='BN1d') -> None:
        super().__init__()
        # first element of layers should be either 3 (for vertices) or 64 (for dt embeddings)
        self.encoder = MLP(layers + [embed_dim], norm_type=norm_type)
        nn.init.constant_(self.encoder[-1].bias, 0.0)
    
    def forward(self, embedding: torch.Tensor):
        """ vertices: [b, N, 3] vertices coordinates with score confidence (x y c)
            distance: [b, N, 64]
        """
        input = embedding.transpose(1, 2) # [b, C, N] C = 3 for vertices, C = 64 for dt
        return self.encoder(input) # [b, 256, N]