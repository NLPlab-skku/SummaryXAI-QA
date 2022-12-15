# -*- coding:utf-8 -*-

import numpy as np

import torch
import torch.nn as nn
import torch.nn.functional as F
import dgl
import dgl.function as fn
from dgl.nn.functional import edge_softmax
        
class GraphEmbedding(nn.Module):
    def __init__(self, cfg):
        super().__init__()
        self.d_model = cfg.d_model
        self.enc_len = cfg.enc_len

    def forward(self, g, token_embedding, token2node):
        '''
            Inputs
                g : DGL Graph
                token_embedding.shape = (B, E_L, H)
                token2node.shape = (B, E_L)
            Outputs
                dgl.graph
        '''
        self.device = g.device
        batch_size = token_embedding.shape[0]
        
        token2node = torch.where(token2node < 0, self.enc_len, token2node)
        _token2node = token2node.unsqueeze(-1).expand_as(token_embedding)

        wnode_embedding = torch.zeros((batch_size, self.enc_len + 1, self.d_model), dtype = torch.float, device = self.device)

        wnode_embedding = wnode_embedding.scatter_add(dim = 1, index = _token2node, src = token_embedding)
        wnode_embedding[:, self.enc_len, :] = 0.

        wnode_division_map = torch.ones_like(token2node, dtype = torch.float)
        wnode_division = torch.zeros((batch_size, wnode_embedding.shape[1]), dtype = torch.float, device = self.device)
        wnode_division = wnode_division.scatter_add(dim = 1, index = token2node, src = wnode_division_map)
        wnode_division = wnode_division.masked_fill(wnode_division == 0, 1).unsqueeze(-1)

        wnode_embedding /= wnode_division

        rand = torch.rand_like(wnode_embedding)
        wnode_embedding = torch.where(wnode_embedding == 0., rand, wnode_embedding)

        output_g = []

        for w_emb, u_g in zip(wnode_embedding, dgl.unbatch(g)):
            num_nodes = u_g.num_nodes()

            _w_emb = w_emb[:num_nodes, :]

            u_g.ndata['feat'] = _w_emb

            output_g.append(u_g)

        return dgl.batch(output_g)

class MkHidden(nn.Module):
    def __init__(self, cfg):
        super().__init__()
        self.cfg = cfg
        self.d_model = cfg.d_model
        
    def forward(self, g):
        self.device = g.device
        batch_size = len(dgl.unbatch(g))
                
        graph_hidden_states = torch.zeros((batch_size, self.cfg.enc_len, self.d_model), dtype = torch.float, device = self.device)
        graph_attention_mask = torch.zeros((batch_size, self.cfg.enc_len), dtype = torch.long, device = self.device)
        for i, u_g in enumerate(dgl.unbatch(g)):
            num_nodes = u_g.num_nodes()

            graph_hidden_states[i, :num_nodes, :] = u_g.ndata['feat']
            graph_attention_mask[i, :num_nodes] = 1

        return graph_hidden_states, graph_attention_mask

class QAGATConv(nn.Module):
    def __init__(self, in_feats, out_feats, num_heads, feat_drop=0., negative_slope=0.2, bias=True):
        super().__init__()
        self._num_heads = num_heads
        self._in_src_feats, self._in_dst_feats = in_feats, in_feats
        self._out_feats = out_feats

        self.fc = nn.Linear(self._in_src_feats, out_feats * num_heads, bias=False)
        self.attn_l = nn.Parameter(torch.FloatTensor(size=(1, num_heads, out_feats)))
        self.attn_r = nn.Parameter(torch.FloatTensor(size=(1, num_heads, out_feats)))

        self.feat_drop = nn.Dropout(feat_drop)
        self.leaky_relu = nn.LeakyReLU(negative_slope)
        if bias:
            self.bias = nn.Parameter(torch.FloatTensor(size=(num_heads * out_feats,)))
        else:
            self.register_buffer('bias', None)
        self.reset_parameters()

    def reset_parameters(self):
        gain = nn.init.calculate_gain('relu')
        if hasattr(self, 'fc'):
            nn.init.xavier_normal_(self.fc.weight, gain=gain)
        else:
            nn.init.xavier_normal_(self.fc_src.weight, gain=gain)
            nn.init.xavier_normal_(self.fc_dst.weight, gain=gain)
        nn.init.xavier_normal_(self.attn_l, gain=gain)
        nn.init.xavier_normal_(self.attn_r, gain=gain)
        if self.bias is not None:
            nn.init.constant_(self.bias, 0)

    def forward(self, graph, feat):
        '''
            Inputs
                graph : DGLGraph
                feat.shape = (# nodes, H)
            Outputs
                rst.shape = (# nodes, # heads, H)
        '''
        with graph.local_scope():
            src_prefix_shape = dst_prefix_shape = feat.shape[:-1]
            # src_prefix_shape, dst_prefix_shape = (# nodes)
            h_src = h_dst = self.feat_drop(feat)
            feat_src = feat_dst = self.fc(h_src).view(*src_prefix_shape, self._num_heads, self._out_feats)
            # feat_src.shape = (# nodes, # heads, H / # heads), feat_dst.shape = (# nodes, # heads, H / # heads)
                
            # Calculate GAT probability
            el = (feat_src * self.attn_l).sum(dim=-1).unsqueeze(-1)
            er = (feat_dst * self.attn_r).sum(dim=-1).unsqueeze(-1)
            # el.shape = (# nodes, # heads, 1), er.shape = (# nodes, # heads, 1)
            graph.srcdata.update({'ft': feat_src, 'el': el})
            graph.dstdata.update({'er': er})
            graph.apply_edges(fn.u_add_v('el', 'er', 'e'))
            e = self.leaky_relu(graph.edata.pop('e'))
            graph.edata['a'] = edge_softmax(graph, e)
            
            # Calculate PageRank probability
            graph.update_all(
                message_func = fn.copy_src(src = 'pr_score', out = 'p'),
                reduce_func = fn.sum(msg = 'p', out = 'sum')
            )
            graph.apply_edges(fn.u_div_v('pr_score', 'sum', 'b'))

            # Calculate Combined probability
            a = graph.edata.pop('a')
            b = graph.edata.pop('b').unsqueeze(-1).unsqueeze(-1)
            graph.edata['c'] = a * b
            
            graph.update_all(
                message_func = fn.copy_edge(edge = 'c', out = 'temp'),
                reduce_func = fn.sum(msg = 'temp', out = 'sum')
            )
            graph.apply_edges(fn.e_div_v('c', 'sum', 'c'))
                        
            # message passing
            graph.update_all(fn.u_mul_e('ft', 'c', 'm'),
                             fn.sum('m', 'ft'))
            rst = graph.dstdata['ft']

            # bias
            if self.bias is not None:
                rst = rst + self.bias.view(
                    *((1,) * len(dst_prefix_shape)), self._num_heads, self._out_feats)

            return rst

class QAGAT(nn.Module):
    def __init__(self, cfg):
        super().__init__()
        self.cfg = cfg
        self.d_model = cfg.d_model
        head_outdim = int(cfg.d_model / 8)
        self.GATLayers = nn.ModuleList([QAGATConv(cfg.d_model, head_outdim, 8, 0.1) for _ in range(cfg.n_gnn - 1)])
        self.last_gatlayer = QAGATConv(cfg.d_model, cfg.d_model, 8, 0.1)

    def forward(self, g):
        '''
            Inputs
                g : dgl.Graph
            Outputs
                graph_hidden_states.shape = (B, E_L, H)
                graph_attention_mask.shape = (B, E_L)
        '''        
        h = g.ndata['feat']

        for gatlayer in self.GATLayers:
            prev_h = h
            temp = gatlayer(g, h)
            h = F.elu(torch.cat([temp[:, i, :] for i in range(temp.shape[1])], dim = -1))
            h = prev_h + h
        
        prev_h = h
        temp = self.last_gatlayer(g, h)
        h = torch.mean(temp, dim = 1)
        h = prev_h + h

        g.ndata['feat'] = h

        return g
