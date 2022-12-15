# -*- coding: utf-8 -*-

from src.models.qagat import GraphEmbedding, QAGAT, MkHidden
import torch
from torch import nn
import torch.nn.functional as F
from transformers import AutoTokenizer

def _expand_mask(mask, tgt_len = None):
    """
        Inputs
            mask.shape = (B, S_L)
        Outputs
            output.shape = (B, 1, T_L, S_L)
    """
    batch_size, src_len = mask.size()
    tgt_len = tgt_len if tgt_len is not None else src_len

    expanded_mask = mask[:, None, None, :].expand(batch_size, 1, tgt_len, src_len).to(torch.float)

    inverted_mask = 1.0 - expanded_mask

    return inverted_mask.masked_fill(inverted_mask.bool(), torch.finfo(torch.float).min)

def _make_causal_mask(dec_ids, past_key_values_length: int = 0):
    """
        Inputs
            dec_ids.shape = (B, D_L) or (B, 1)
    """
    batch_size, tgt_len = dec_ids.size()
    device = dec_ids.device

    mask = torch.full((tgt_len, tgt_len), float("-inf"))
    mask_cond = torch.arange(mask.size(-1))
    mask.masked_fill_(mask_cond < (mask_cond + 1).view(mask.size(-1), 1), 0)
    mask = mask.to(torch.float).to(device)

    if past_key_values_length > 0:
        mask = torch.cat([torch.zeros(tgt_len, past_key_values_length, dtype=torch.float, device = device), mask], dim=-1)
    return mask[None, None, :, :].expand(batch_size, 1, tgt_len, tgt_len + past_key_values_length)

class PositionalEncoding(nn.Module):
    def __init__(self, d_model):
        super().__init__()
        self.encoding = torch.zeros(1024, d_model)
        self.encoding.requires_grad = False

        pos = torch.arange(0, 1024)
        pos = pos.float().unsqueeze(dim = 1)

        _2i = torch.arange(0, d_model, step = 2).float()

        self.encoding[:, ::2] = torch.sin(pos / (10000 ** (_2i / d_model)))
        self.encoding[:, 1::2] = torch.cos(pos / (10000 ** (_2i / d_model)))

    def forward(self, x, past_key_values_length = 0):
        batch_size, seq_len = x.size()
        device = x.device

        if past_key_values_length == 0:
            return self.encoding[:seq_len, :].unsqueeze(0).to(device)
        else:
            return self.encoding[past_key_values_length, :].unsqueeze(0).unsqueeze(0).to(device)

class MultiHeadAttention(nn.Module):
    def __init__(self, cfg, is_decoder = False, is_cross_attention = False):
        super().__init__()
        self.d_model = cfg.d_model
        self.num_heads = cfg.num_heads
        self.attention_dropout = cfg.attention_dropout
        self.d_head = self.d_model // self.num_heads
        self.scaling = self.d_head ** -0.5
        self.is_decoder = is_decoder
        self.is_cross_attention = is_cross_attention

        self.q_proj = nn.Linear(self.d_model, self.d_model)
        self.k_proj = nn.Linear(self.d_model, self.d_model)
        self.v_proj = nn.Linear(self.d_model, self.d_model)
        self.out_proj = nn.Linear(self.d_model, self.d_model)

    def _shape(self, tensor, seq_len, batch_size):
        return tensor.view(batch_size, seq_len, self.num_heads, self.d_head).transpose(1, 2).contiguous()

    def forward(self, query_states, key_value_states, past_key_value = None, attention_mask = None):
        '''
            Inputs
                query_states.shape = (B, T_L, H)
                key_value_states.shape = (B, S_L, H)
                past_key_value.shape = ((B, num_heads, S_L, H // num_heads), (B, num_heads, S_L, H // num_heads))
                attention_mask.shape = (B, 1, T_L, S_L)
        '''
        batch_size, tgt_len, d_model = query_states.size()
        _, src_len, _ = key_value_states.size()

        query_states = self._shape(self.q_proj(query_states) * self.scaling, tgt_len, batch_size)
        if self.is_cross_attention and past_key_value is not None:
            # Encoder key, value
            key_states = past_key_value[0]
            value_states = past_key_value[1]
        elif self.is_cross_attention:
            key_states = self._shape(self.k_proj(key_value_states), -1, batch_size)
            value_states = self._shape(self.v_proj(key_value_states), -1, batch_size)
        elif past_key_value is not None:
            key_states = self._shape(self.k_proj(key_value_states), -1, batch_size)
            value_states = self._shape(self.v_proj(key_value_states), -1, batch_size)
            key_states = torch.cat([past_key_value[0], key_states], dim = 2)
            value_states = torch.cat([past_key_value[1], value_states], dim = 2)
        else:
            key_states = self._shape(self.k_proj(key_value_states), -1, batch_size)
            value_states = self._shape(self.v_proj(key_value_states), -1, batch_size)

        if self.is_decoder:
            past_key_value = (key_states, value_states)

        proj_shape = (batch_size * self.num_heads, -1, self.d_head)
        query_states, key_states, value_states = query_states.view(*proj_shape), key_states.view(*proj_shape), value_states.view(*proj_shape)
        # query_states.shape = (B * num_heads, T_L, H // num_heads), key_states.shape = (B * num_heads, S_L, H // num_heads)

        attn_weights = torch.matmul(query_states, key_states.transpose(1, 2))
        # attn_weights.shape = (B * num_heads, T_L, S_L)
        
        if attention_mask is not None:
            attn_weights = attn_weights.view(batch_size, self.num_heads, tgt_len, src_len) + attention_mask
            attn_weights = attn_weights.view(batch_size * self.num_heads, tgt_len, src_len)
        attn_weights = F.softmax(attn_weights, dim = -1)

        attn_probs = F.dropout(attn_weights, p = self.attention_dropout, training = self.training)

        attn_output = torch.matmul(attn_probs, value_states)
        attn_output = attn_output.view(batch_size, self.num_heads, tgt_len, self.d_head)
        attn_output = attn_output.transpose(1, 2)
        attn_output = attn_output.reshape(batch_size, tgt_len, d_model)

        attn_output = self.out_proj(attn_output)

        return attn_output, past_key_value

class EncoderLayer(nn.Module):
    def __init__(self, cfg):
        super().__init__()
        self.d_model = cfg.d_model

        self.self_attn = MultiHeadAttention(cfg)
        self.self_attn_layer_norm = nn.LayerNorm(self.d_model)

        self.dropout = cfg.dropout
        self.activation_fn = nn.ReLU()
        self.activation_dropout = cfg.activation_dropout

        self.fc1 = nn.Linear(self.d_model, 4 * self.d_model)
        self.fc2 = nn.Linear(4 * self.d_model, self.d_model)
        self.final_layer_norm = nn.LayerNorm(self.d_model)

    def forward(self, hidden_states, enc_self_mask):
        '''
            Inputs
                hidden_states.shape = (B, L, H)
                attention_mask.shape = (B, 1, E_L, E_L)
        '''
        residual = hidden_states
        hidden_states, _ = self.self_attn(
            query_states = hidden_states,
            key_value_states = hidden_states,
            attention_mask = enc_self_mask
        )
        hidden_states = F.dropout(hidden_states, p = self.dropout, training = self.training)
        hidden_states = residual + hidden_states
        hidden_states = self.self_attn_layer_norm(hidden_states)

        residual = hidden_states
        hidden_states = self.activation_fn(self.fc1(hidden_states))
        hidden_states = F.dropout(hidden_states, p=self.activation_dropout, training=self.training)
        hidden_states = self.fc2(hidden_states)
        hidden_states = F.dropout(hidden_states, p=self.dropout, training=self.training)
        hidden_states = residual + hidden_states
        hidden_states = self.final_layer_norm(hidden_states)

        return (hidden_states, )

class DecoderLayer(nn.Module):
    def __init__(self, cfg):
        super().__init__()
        self.d_model = cfg.d_model

        self.self_attn = MultiHeadAttention(cfg, is_decoder = True, is_cross_attention = False)
        self.self_attn_layer_norm = nn.LayerNorm(self.d_model)

        self.dropout = cfg.dropout
        self.activation_fn = nn.ReLU()
        self.activation_dropout = cfg.activation_dropout

        self.cross_attn = MultiHeadAttention(cfg, is_decoder = True, is_cross_attention = True)
        self.cross_attn_layer_norm = nn.LayerNorm(self.d_model)

        self.graph_attn = MultiHeadAttention(cfg, is_decoder = True, is_cross_attention = True)
        self.graph_attn_layer_norm = nn.LayerNorm(self.d_model)

        self.fc1 = nn.Linear(self.d_model, 4 * self.d_model)
        self.fc2 = nn.Linear(4 * self.d_model, self.d_model)
        self.final_layer_norm = nn.LayerNorm(self.d_model)

    def forward(
        self, 
        hidden_states, 
        dec_self_mask = None, 
        enc_hidden_states = None, 
        enc_dec_mask = None, 
        graph_hidden_states = None,
        graph_dec_mask = None,
        past_key_value = None
    ):
        '''
            Inputs
                hidden_states.shape = (B, D_L, H),
                dec_self_mask.shape = (B, 1, D_L, D_L),
                enc_hidden_states.shape = (B, E_L, H),
                enc_dec_mask.shape = (B, 1, D_L, E_L),
                graph_hidden_states.shape = (B, E_L, H),
                graph_dec_mask.shape = (B, 1, D_L, E_L)
                past_key_value = Tuple
        '''
        self_attn_past_key_value = past_key_value[:2] if past_key_value is not None else None
        cross_attn_past_key_value = past_key_value[2:4] if past_key_value is not None else None
        graph_attn_past_key_value = past_key_value[4:] if past_key_value is not None else None

        residual = hidden_states
        hidden_states, self_attn_present_key_value = self.self_attn(
            query_states = hidden_states,
            key_value_states = hidden_states,
            past_key_value = self_attn_past_key_value,
            attention_mask = dec_self_mask
        )
        hidden_states = F.dropout(hidden_states, p=self.dropout, training=self.training)
        hidden_states = residual + hidden_states
        hidden_states = self.self_attn_layer_norm(hidden_states)

        residual = hidden_states

        _enc_hidden_states, cross_attn_present_key_value = self.cross_attn(
            query_states = hidden_states,
            key_value_states = enc_hidden_states,
            past_key_value = cross_attn_past_key_value,
            attention_mask = enc_dec_mask,
        )
        _enc_hidden_states = F.dropout(_enc_hidden_states, p = self.dropout, training = self.training)
        _enc_hidden_states = self.cross_attn_layer_norm(_enc_hidden_states)

        _graph_hidden_states, graph_attn_present_key_value = self.graph_attn(
            query_states = hidden_states,
            key_value_states = graph_hidden_states,
            past_key_value = graph_attn_past_key_value,
            attention_mask = graph_dec_mask
        )
        _graph_hidden_states = F.dropout(_graph_hidden_states, p = self.dropout, training = self.training)
        _graph_hidden_states = F.dropout(_graph_hidden_states, p = self.dropout, training = self.training)

        hidden_states = residual + _enc_hidden_states + _graph_hidden_states

        residual = hidden_states
        hidden_states = self.activation_fn(self.fc1(hidden_states))
        hidden_states = F.dropout(hidden_states, p = self.activation_dropout, training = self.training)
        hidden_states = self.fc2(hidden_states)
        hidden_states = F.dropout(hidden_states, p=self.dropout, training=self.training)
        hidden_states = residual + hidden_states
        hidden_states = self.final_layer_norm(hidden_states)
        
        present_key_value = self_attn_present_key_value + cross_attn_present_key_value + graph_attn_present_key_value

        return (hidden_states, present_key_value)

class Encoder(nn.Module):
    def __init__(self, cfg, embed_tokens, embed_positions):
        super().__init__()
        self.d_model = cfg.d_model
        self.dropout = cfg.dropout

        self.embed_tokens = embed_tokens
        self.embed_positions = embed_positions

        self.layers = nn.ModuleList([EncoderLayer(cfg) for _ in range(cfg.num_encoder_layers)])
        self.embedding_layer_norm = nn.LayerNorm(self.d_model)

    def forward(self, enc_ids, enc_mask):
        '''
            Inputs
                enc_ids.shape = (B, E_L)
                enc_mask.shape = (B, E_L)
            Outputs
                output.shape = (B, E_L, H)
        '''
        token_embedding = self.embed_tokens(enc_ids)
        pos_embedding = self.embed_positions(enc_ids)
        hidden_states = token_embedding + pos_embedding
        hidden_states = self.embedding_layer_norm(hidden_states)
        hidden_states = F.dropout(hidden_states, p = self.dropout, training = self.training)

        enc_self_mask = _expand_mask(enc_mask)
        # enc_self_mask.shape = (B, 1, E_L, E_L)

        for encoder_layer in self.layers:
            layer_outputs = encoder_layer(hidden_states, enc_self_mask)
            hidden_states = layer_outputs[0]

        return {
            'enc_hidden_states' : hidden_states
        }

class Decoder(nn.Module):
    def __init__(self, cfg, embed_tokens, embed_positions):
        super().__init__()
        self.d_model = cfg.d_model
        self.dropout = cfg.dropout

        self.embed_tokens = embed_tokens
        self.embed_positions = embed_positions

        self.layers = nn.ModuleList([DecoderLayer(cfg) for _ in range(cfg.num_decoder_layers)])
        self.embedding_layer_norm = nn.LayerNorm(self.d_model)

    def forward(
        self, 
        dec_ids,
        dec_mask = None,
        enc_hidden_states = None,
        enc_mask = None,
        graph_hidden_states = None,
        graph_mask = None,
        past_key_values = None
    ):
        '''
            Inputs
                Case 1 : Training
                    dec_ids.shape = (B, D_L)
                    dec_mask.shape = (B, D_L)
                    enc_hidden_states.shape = (B, E_L, H)
                    enc_mask.shape = (B, E_L)
                    graph_hidden_states.shape = (B, E_L, H)
                    graph_mask.shape = (B, E_L)
                Case 2 : Inference
                    dec_ids.shape = (B, 1)
                    enc_hidden_states.shape = (B, E_L, H)
                    enc_mask.shape = (B, E_L)
                    graph_hidden_states.shape = (B, E_L, H)
                    graph_mask.shape = (B, E_L)
                    past_key_values = (self_attn_key_value, cross_attn_key_value, graph_attn_key_value)

            Outputs
                hidden_states.shape = (B, D_L, H) or (B, 1, H)
                past_key_values = ((layer1 self_attn_key, layer1 self_attn_value, layer1 cross_attn_key, layer1 cross_attn_value), ... )
        '''
        past_key_values_length = past_key_values[0][0].shape[2] if past_key_values is not None else 0
        token_embedding = self.embed_tokens(dec_ids)
        pos_embedding = self.embed_positions(dec_ids, past_key_values_length)

        hidden_states = token_embedding + pos_embedding
        hidden_states = self.embedding_layer_norm(hidden_states)
        hidden_states = F.dropout(hidden_states, p = self.dropout, training = self.training)

        dec_self_mask = None
        if dec_mask is not None:
            temp1 = _make_causal_mask(dec_ids)
            temp2 = _expand_mask(dec_mask)
            dec_self_mask = temp1 + temp2
        enc_dec_mask = _expand_mask(enc_mask, dec_ids.shape[-1])
        graph_dec_mask = _expand_mask(graph_mask, dec_ids.shape[-1])

        cache = ()

        for idx, decoder_layer in enumerate(self.layers):
            past_key_value = past_key_values[idx] if past_key_values is not None else None

            layer_outputs = decoder_layer(hidden_states, dec_self_mask, enc_hidden_states, enc_dec_mask, graph_hidden_states, graph_dec_mask, past_key_value)

            hidden_states = layer_outputs[0]
            cache += (layer_outputs[1],)

        past_key_values = cache

        return {
            'dec_hidden_states' : hidden_states,
            'past_key_values' : past_key_values
        }

class QSGTransformer(nn.Module):
    def __init__(self, cfg):
        super().__init__()
        self.embed_tokens = nn.Embedding(cfg.vocab_size, cfg.d_model, cfg.pad_token_id)
        self.embed_positions = PositionalEncoding(cfg.d_model)

        self.encoder = Encoder(cfg, self.embed_tokens, self.embed_positions)

        self.graph_embedding = GraphEmbedding(cfg)
        self.qagat = QAGAT(cfg)
        self.mk_hidden = MkHidden(cfg)

        self.decoder = Decoder(cfg, self.embed_tokens, self.embed_positions)

    def forward(
        self,
        enc_ids,
        enc_mask,
        token2node,
        g,
        dec_ids,
        dec_mask = None,
        enc_hidden_states = None,
        graph_hidden_states = None,
        graph_mask = None,
        past_key_values = None
    ):

        if enc_hidden_states is None and graph_hidden_states is None:
            enc_outputs = self.encoder(enc_ids, enc_mask)
            enc_hidden_states = enc_outputs['enc_hidden_states']
            g = self.graph_embedding(g, enc_hidden_states, token2node)
            g = self.qagat(g)
            graph_hidden_states, graph_mask = self.mk_hidden(g)

        dec_outputs = self.decoder(dec_ids, dec_mask, enc_hidden_states, enc_mask, graph_hidden_states, graph_mask, past_key_values)

        return {
            'enc_hidden_states' : enc_hidden_states,
            'graph_hidden_states' : graph_hidden_states,
            'graph_mask' : graph_mask,
            'dec_hidden_states' : dec_outputs['dec_hidden_states'],
            'past_key_values' : dec_outputs['past_key_values']
        }

class QSGTransformerForConditionalGeneration(nn.Module):
    def __init__(self, cfg):
        super().__init__()
        self.cfg = cfg
        self.tokenizer = AutoTokenizer.from_pretrained(cfg.model_name)
        self.model = QSGTransformer(cfg)
        self.lm_head = nn.Linear(cfg.d_model, cfg.vocab_size, bias = False)
        self.aux_head = nn.Linear(cfg.d_model, 6)

    def forward(
        self,
        enc_ids,
        enc_mask,
        token2node,
        g,
        dec_ids,
        dec_mask = None,
        enc_hidden_states = None,
        graph_hidden_states = None,
        graph_mask = None,
        past_key_values = None,
        label_ids = None,
        aux_label = None,
    ):
        model_outputs = self.model(
            enc_ids, enc_mask, token2node, g, dec_ids, dec_mask, enc_hidden_states, graph_hidden_states, graph_mask, past_key_values
        )
        lm_logits = self.lm_head(model_outputs['dec_hidden_states'])
        aux_logits = self.aux_head(model_outputs['graph_hidden_states'])

        lm_loss = None
        if label_ids is not None:
            criterion = nn.CrossEntropyLoss()
            lm_loss = criterion(lm_logits.view(-1, self.cfg.vocab_size), label_ids.view(-1))

        aux_loss = None
        if aux_label is not None:
            criterion = nn.CrossEntropyLoss()
            aux_loss = criterion(aux_logits.view(-1, 6), aux_label.view(-1))

        loss = None
        if lm_loss is not None and aux_loss is not None:
            loss = (1 - self.cfg.aux_rate) * lm_loss + self.cfg.aux_rate * aux_loss
        elif lm_loss is not None and aux_loss is None:
            loss = lm_loss

        return {
            'enc_hidden_states' : model_outputs['enc_hidden_states'],
            'graph_hidden_states' : model_outputs['graph_hidden_states'],
            'graph_mask' : model_outputs['graph_mask'],
            'dec_hidden_states' : model_outputs['dec_hidden_states'],
            'past_key_values' : model_outputs['past_key_values'],
            'lm_logits' : lm_logits,
            'lm_loss' : lm_loss,
            'aux_loss' : aux_loss,
            'loss' : loss
        }

    def inference(
        self,
        enc_ids,
        enc_mask,
        token2node,
        g,    
    ):
        batch_size = enc_ids.shape[0]
        bos_token_id = self.tokenizer.bos_token_id
        eos_token_id = self.tokenizer.eos_token_id

        outputs = []
        has_eos = torch.zeros(batch_size, dtype = torch.bool).to(enc_ids.device)
        
        dec_ids = torch.tensor([[bos_token_id]] * batch_size, dtype = torch.long, device = enc_ids.device)
        dec_mask = None
        enc_hidden_states = None
        graph_hidden_states = None
        graph_mask = None
        past_key_values = None

        for _ in range(self.cfg.dec_len):
            model_outputs = self.forward(
                enc_ids, enc_mask, token2node, g, dec_ids, dec_mask, enc_hidden_states, graph_hidden_states, graph_mask, past_key_values
            )

            new_token_ids = torch.argmax(model_outputs['lm_logits'][:, -1, :], dim = -1)

            has_eos = has_eos | (new_token_ids == eos_token_id)
            new_token_ids = new_token_ids.masked_fill(has_eos, eos_token_id)
            outputs.append(new_token_ids)

            dec_ids = new_token_ids.unsqueeze(-1)
            enc_hidden_states = model_outputs['enc_hidden_states']
            graph_hidden_states = model_outputs['graph_hidden_states']
            graph_mask = model_outputs['graph_mask']
            past_key_values = model_outputs['past_key_values']

            if torch.all(has_eos):
                break
            
        outputs = torch.stack(outputs, dim = -1).tolist()
        generated_summary = []

        for output in outputs:
            generated_summary.append(self.tokenizer.decode(output, skip_special_tokens = True))

        return {
            'generated_summary' : generated_summary
        }