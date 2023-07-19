import torch
from einops import rearrange
from copy import deepcopy
from mmcv.cnn.bricks.registry import PLUGIN_LAYERS, TRANSFORMER_LAYER_SEQUENCE
# from mmcv.cnn.bricks.transformer import TransformerLayerSequence
from mmdet3d.models.builder import build_backbone
from mmcv.cnn import build_conv_layer, build_plugin_layer, build_upsample_layer
from mmcv.cnn.bricks.transformer import build_feedforward_network, build_transformer_layer_sequence
from mmcv.runner.base_module import BaseModule, ModuleList, Sequential
from mmdet.models.utils.transformer import inverse_sigmoid

if __package__ == '':
    import sys
    from os import path
    print(path.dirname(path.dirname(path.abspath(__file__))))
    sys.path.append(path.dirname(path.dirname(path.abspath(__file__))))
    from graph import *
else:
    from .graph import *

@PLUGIN_LAYERS.register_module()
class ElementDecoder(BaseModule):
    """Implements the decoder in DETR3D transformer.
    Args:
        return_intermediate (bool): Whether to return intermediate outputs.
        coder_norm_cfg (dict): Config of last normalization layer. Default：
            `LN`.
    """

    def __init__(self, bev_encoder, conv_cfg, bev_decoder=None):
        super(ElementDecoder, self).__init__()
        self.fp16_enabled = False
        self.bev_encoder = build_backbone(bev_encoder)
        self.conv = build_conv_layer(conv_cfg)
        
        if bev_decoder is not None:
            in_channels = bev_decoder['in_channels']
            upblocks = []
            for i, out_channels in enumerate(bev_decoder['blocks']):
                up = build_upsample_layer(bev_decoder['upsample_cfg'])
                conv_module = build_plugin_layer(
                    bev_decoder['plugin_cfg'], 
                    in_channels=in_channels, 
                    out_channels=out_channels)[1]
                upblock = Sequential(
                    up,
                    conv_module,
                )
                upblocks.append(upblock)
                in_channels = out_channels
            self.upblocks = ModuleList(upblocks)
        else:
            self.upblocks = None

    def forward(self,
                bev_embed, # [bs, bev_h, bev_w, embed_dims=256]
                ):
        """Forward function for `Detr3DTransformerDecoder`.
        Args:
            bev_embed (Tensor): Input query with shape
                `(num_query, bs, embed_dims)`.
            reference_points (Tensor): The reference
                points of offset. has shape
                (bs, num_query, 4) when as_two_stage,
                otherwise has shape ((bs, num_query, 2).
            reg_branch: (obj:`nn.ModuleList`): Used for
                refining the regression results. Only would
                be passed when with_box_refine is True,
                otherwise would be passed a `None`.
        Returns:
            Tensor: Results with shape [1, num_query, bs, embed_dims] when
                return_intermediate is `False`, otherwise it has shape
                [num_layers, num_query, bs, embed_dims].
        """
        bev_embed = self.bev_encoder(bev_embed)
        bev_embed = bev_embed[0]
        if self.upblocks is not None:
            for upblock in self.upblocks:
                bev_embed = upblock(bev_embed)
        output = self.conv(bev_embed)

        return output
    
@TRANSFORMER_LAYER_SEQUENCE.register_module()
class InstaGraMDecoder(BaseModule):
    """Implements the decoder in DETR3D transformer.
    Args:
        return_intermediate (bool): Whether to return intermediate outputs.
        coder_norm_cfg (dict): Config of last normalization layer. Default：
            `LN`.
    """
    
    def __init__(self, 
                 vertex_decoder: dict,
                 distance_decoder: dict,
                 graph_encoder: dict,
                 attn_cfg: dict,
                 cell_size,
                 use_dist_embed,
                #  pc_range,
                #  voxel_size,
                 dist_thr,
                 vertex_thr,
                 num_vertices,
                 embed_dim,
                 pos_freq,
                 sinkhorn_iters,
                 num_gnn_layers,
                 ):
        super(InstaGraMDecoder, self).__init__()
        
        self.cell_size = cell_size
        self.use_dist_embed = use_dist_embed
        self.dist_thr = dist_thr
        self.vertex_thr = vertex_thr
        self.num_vertices = num_vertices
        self.embed_dim = embed_dim
        self.sinkhorn_iters = sinkhorn_iters
        self.num_layers = num_gnn_layers
        
        self.vertex_decoder = build_plugin_layer(vertex_decoder)[1]
        self.distance_decoder = build_plugin_layer(distance_decoder)[1]
        
        # Graph embedding
        dist_embed_dim = cell_size*cell_size if use_dist_embed \
            else distance_decoder['conv_cfg']['out_channels']
        self.pe_fn, pe_dim = get_embedder(pos_freq)
        positional_graph_enc = deepcopy(graph_encoder)
        positional_graph_enc['layers'] = [pe_dim+1, *graph_encoder['layers']]
        self.venc = build_feedforward_network(positional_graph_enc)
        directional_graph_enc = deepcopy(graph_encoder)
        directional_graph_enc['layers'] = [dist_embed_dim, *graph_encoder['layers']]
        self.denc = build_feedforward_network(directional_graph_enc)
        # Attentional GNN
        self.gnn = build_transformer_layer_sequence(
            attn_cfg,
            default_args=dict(layer_names=['self']*num_gnn_layers))
        self.final_proj = build_conv_layer(
            cfg=dict(type='Conv1d',
            in_channels=embed_dim,
            out_channels=embed_dim,
            kernel_size=1,
            bias=True
            )
        )
        
        # Bin score for adj. matrix
        bin_score = torch.nn.Parameter(torch.tensor(1.))
        self.register_parameter('bin_score', bin_score)
        
        # Vertex offset regressor
        self.offset_head = build_conv_layer(
            cfg=dict(type='Conv1d',
            in_channels=embed_dim,
            out_channels=2,
            kernel_size=1,
            bias=True)
        )
        
    def forward(self, 
                bev_embed, # b, 256, 200, 100
                # vertex: torch.Tensor,   # b, 3, 50, 25
                # distance: torch.Tensor  # b, 3, 400, 200
                ):
        vertex = self.vertex_decoder(bev_embed)     # b, 3, 50, 25
        distance = self.distance_decoder(bev_embed) # b, 3, 400, 200
        
        scores = vertex.sigmoid()
        b_org, c_org, _, _ = scores.shape
        scores = rearrange(scores, 'b c ... -> (b c) ...') # (b c) 25 50
        score_shape = scores.shape # (b c) 25 50
        
        # [2] Extract vertices using NMS
        vertices = [torch.nonzero(s > self.vertex_thr) for s in scores] # list of length (b c), [N, 2(row, col)] tensor
        scores = [s[tuple(v.t())] for s, v in zip(scores, vertices)] # list of length (b c), [N] tensor
        # vertices_cell = [(v / self.cell_size).trunc().long() for v in vertices]

        # Extract distance transform
        if self.use_dist_embed:
            distance = rearrange(distance, 'b c ... -> (b c) ...') # (b c) 200 400
            dt_embedding = sample_dt(
                vertices, 
                distance.sigmoid(), # F.relu(distance).clamp(max=self.dist_thr) 
                self.cell_size) # list of [N, 64] tensor
        else:
            # distance: feature [b, 256, 100, 200]
            bf, cf, hf, wf = distance.shape
            distance = distance.view(bf, 1, cf, hf, wf).repeat(1, c_org, 1, 1, 1) # b c 256 100 200
            distance = rearrange(distance, 'b c ... -> (b c) ...') # (b c) 256 100 200
            distance_down = F.interpolate(
                distance, 
                scale_factor=0.25, 
                mode='bilinear', 
                align_corners=True) # (b c) 256 25 50
            dt_embedding = sample_feat(vertices, distance_down) # list of [N, 256] tensor

        if self.num_vertices >= 0:
            vertices, scores, dt_embedding, masks = list(zip(*[
                top_k_vertices(v, s, d, self.num_vertices) # v: N, 2
                for v, s, d in zip(vertices, scores, dt_embedding)
            ]))

        # Convert (h, w) to (x, y), normalized
        # v: [N, 2]
        vertices_norm = [
            normalize_vertices(torch.flip(v, [1]).float(), score_shape) for v in vertices
            ] # (b c) list of [N, 2] tensor

        # Vertices in pixel coordinate
        vertices = torch.stack(vertices).flip([2]) # (b c) N 2; x: [0~49], y: [0~24]

        # Positional embedding (x, y, c)
        pos_embedding = [
            torch.cat(
                (self.pe_fn(v), s.unsqueeze(1)), 1) for v, s in zip(vertices_norm, scores)
            ] # (b c) list of [N, pe_dim+1] tensor
        pos_embedding = torch.stack(pos_embedding) # (b c) N pe_dim+1

        dt_embedding = torch.stack(dt_embedding) # (b c) N 64
        masks = torch.stack(masks).unsqueeze(-1) # (b c) N 1

        graph_embedding = self.venc(pos_embedding) + self.denc(dt_embedding) # for visual descriptor
        graph_embedding = self.gnn(graph_embedding, masks.transpose(1, 2)) # (b c) 256 N, (b c) L 4 N N
        offset = torch.sigmoid(self.offset_head(graph_embedding)) # (b c) 2 N [0, 1]
        graph_embedding = self.final_proj(graph_embedding) # (b c) 256 N

        # Adjacency matrix score as inner product of all nodes
        matches = torch.einsum('bdn,bdm->bnm', graph_embedding, graph_embedding)
        matches = matches / self.embed_dim**.5 # (b c) N N [match.fill_diagonal_(0.0) for match in matches]

        # Don't care self matches
        b, m, n = matches.shape
        diag_mask = torch.eye(m).repeat(b, 1, 1).bool()
        matches[diag_mask] = -1e9

        # Don't care bin matches
        match_mask = torch.einsum('bnd,bmd->bnm', masks, masks) # (b c) N N
        matches = matches.masked_fill(match_mask == 0, -1e9)

        # Matching layer
        if self.sinkhorn_iters > 0:
            matches = log_optimal_transport(matches, self.bin_score, self.sinkhorn_iters) # (b c) N+1 N+1
        else:
            bins0 = self.bin_score.expand(b, m, 1) # (b c) N 1
            bins1 = self.bin_score.expand(b, 1, n) # (b c) 1 N
            alpha = self.bin_score.expand(b, 1, 1) # (b c) 1 1
            matches = torch.cat( # (b c) N+1 N+1
            [
                torch.cat([matches, bins0], -1), # (b c) N N+1
                torch.cat([bins1, alpha], -1)   # (b c) 1 N+1
            ], 1)
            matches = F.log_softmax(matches, -1) # (b c) N+1 N+1
        # matches.exp() should be probability

        # Refinement offset in pixel coordinate
        _, h, w = distance.shape # (b c) 200 400
        offset = offset.permute(0, 2, 1) # (b c) N 2 [0, 1]
        grid_num = vertices.new_tensor([score_shape[2], score_shape[1]]) # 50 25
        vertices = (vertices + offset) / grid_num # [0, 1] normalized
        
        # Concat vertices and scores just for vectorization
        scores = torch.stack(scores).unsqueeze(-1) # (b c) N 1
        vertices = torch.cat([vertices, scores], -1) # (b c) N 3
        
        distance = rearrange(distance, '(b c) ... -> b c ...', b=b_org, c=c_org)
        matches = rearrange(matches, '(b c) ... -> b c ...', b=b_org, c=c_org)
        vertices = rearrange(vertices, '(b c) ... -> b c ...', b=b_org, c=c_org)
        masks = rearrange(masks, '(b c) ... -> b c ...', b=b_org, c=c_org)
        
        return vertex, distance, matches, vertices, masks


if __name__ == '__main__':
    num_map_classes = 3
    _dim_ = 256
    _pos_dim_ = _dim_//2
    use_dist_embed = True
    inputs = torch.rand(1, 256, 200, 100)
    decoder=dict(
        type='InstaGraMDecoder',
        vertex_decoder=dict(
            type='ElementDecoder',
            bev_encoder=dict(
                type='ResNet',
                depth=18,
                in_channels=_dim_,
                base_channels=_pos_dim_,
                num_stages=1,
                strides=(1,),
                dilations=(1,),
                out_indices=(0,),
                ),
            conv_cfg=dict(
                type='Conv2d',
                in_channels=_pos_dim_,
                out_channels=num_map_classes,
                kernel_size=1,
                stride=1,
                padding=0,
            ),
            ),
        distance_decoder=dict(
            type='ElementDecoder',
            bev_encoder=dict(
                type='ResNet',
                depth=18,
                in_channels=_dim_,
                base_channels=_pos_dim_,
                num_stages=1,
                strides=(1,),
                dilations=(1,),
                out_indices=(0,),
                ),
            conv_cfg=dict(
                type='Conv2d',
                in_channels=_pos_dim_,
                out_channels=num_map_classes if use_dist_embed else _dim_,
                kernel_size=1,
                stride=1,
                padding=0,
            ),
            bev_decoder=dict(
                upsample_cfg=dict(
                    type='bilinear',
                    scale_factor=2,
                    align_corners=True,
                    ),
                plugin_cfg=dict(
                    type='ConvModule',
                    kernel_size=3,
                    padding=1,
                    bias=False,
                    norm_cfg=dict(type='BN'),
                    act_cfg=dict(type='ReLU', inplace=True),
                ),
                in_channels=_pos_dim_,
                blocks=(128, 128, _pos_dim_),
            ) if use_dist_embed else None,
            ),
        graph_encoder=dict(
            type='GraphEncoder',
            embed_dim=_dim_,
            layers=[_pos_dim_//2, _pos_dim_, _dim_],
            norm_type='BN1d',
            ),
        attn_cfg=dict(
            type='AttentionalGNN',
            embed_dim=_dim_,
            norm_type='BN1d',
        ),
        cell_size=8,
        use_dist_embed=use_dist_embed,
        # pc_range=point_cloud_range,
        # voxel_size=voxel_size,
        dist_thr=10.0,
        vertex_thr=0.3,
        num_vertices=250,
        embed_dim=_dim_,
        pos_freq=10,
        sinkhorn_iters=100,
        num_gnn_layers=9,
        ) # decoder
    
    # vertex_decoder = build_plugin_layer(decoder['vertex_decoder'])[1]
    # distance_decoder = build_plugin_layer(decoder['distance_decoder'])[1]
    
    # vertex = vertex_decoder(inputs)
    # distance = distance_decoder(inputs)
    
    # print(f'vertex map shape: {tuple(vertex.shape)}')
    # print(f'distance map shape: {tuple(distance.shape)}')
    
    decoder = build_transformer_layer_sequence(decoder)
    
    outputs = decoder(inputs)