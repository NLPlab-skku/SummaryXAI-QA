# -*- coding: utf-8 -*-"""

import torch
from torch import nn
import numpy as np
from transformers import  BartForConditionalGeneration, AutoConfig
import pdb

class Model(nn.Module):
    def __init__(self, args):
        super().__init__()
        # self.tokenizer = AutoTokenizer.from_pretrained(args.model_name)
        self.match_n_layer = args.layers  #12
        self.match_n_head = args.n_heads #16
        self.n_embd = args.d_model #1024
        self.match_n_embd = self.n_embd // self.match_n_head # 64
        self.preseqlen = args.preseqlen
        self.mid_dim = args.mid_dim

        self.model_name = args.model_name
        self.device = args.device
        self.dropout = nn.Dropout(args.dropout)
        self.config = AutoConfig.from_pretrained(self.model_name)
        self.config.preseqlen = self.preseqlen
        self.config.use_prefix = True
        self.bart = BartForConditionalGeneration.from_pretrained(self.model_name,config=self.config)

        self.input_tokens = torch.arange(self.preseqlen).long()

        self.wte = nn.Embedding(self.preseqlen, self.n_embd)
        self.control_trans = nn.Sequential(
            nn.Linear(self.n_embd, self.mid_dim),
            nn.Tanh(),
            nn.Linear(self.mid_dim, self.match_n_layer * 2 * self.n_embd))

        self.wte_enc = nn.Embedding(self.preseqlen, self.n_embd)
        self.control_trans_enc = nn.Sequential(
            nn.Linear(self.n_embd, self.mid_dim),
            nn.Tanh(),
            nn.Linear(self.mid_dim, self.match_n_layer * 2 * self.n_embd))

        self.wte2 = nn.Embedding(self.preseqlen, self.n_embd)
        self.control_trans2 = nn.Sequential(
            nn.Linear(self.n_embd, self.mid_dim),
            nn.Tanh(),
            nn.Linear(self.mid_dim, self.match_n_layer * 2 * self.n_embd))

        self.w_de = nn.Linear(self.n_embd,1,bias=True)

    def get_prompt(self, bsz=None, sample_size=1):#control_code=None, gpt2=None,
        old_bsz = bsz
        bsz = bsz * sample_size
        input_tokens = self.input_tokens.unsqueeze(0).expand(bsz, -1).to(self.device)  #[4,200]
        temp_control = self.wte(input_tokens) #[4,200,1024]
        past_key_values = self.control_trans(temp_control) #bsz, seqlen, layer*emb #[4,200,24576]


        bsz, seqlen, _ = past_key_values.shape
        past_key_values = past_key_values.view(bsz, seqlen, self.match_n_layer * 2, self.match_n_head,
                                               self.match_n_embd) #[4,200,24,16,64]
        past_key_values = self.dropout(past_key_values)
        past_key_values = past_key_values.permute([2, 0, 3, 1, 4]).split(2) #[2,4,16,200,64]


        temp_control2 = self.wte2(input_tokens)
        past_key_values2 = self.control_trans2(temp_control2)  # bsz, seqlen, layer*emb
        bsz, seqlen, _ = past_key_values2.shape
        past_key_values2 = past_key_values2.view(bsz, seqlen, self.match_n_layer * 2, self.match_n_head,
                                               self.match_n_embd)
        past_key_values2 = self.dropout(past_key_values2)
        past_key_values2 = past_key_values2.permute([2, 0, 3, 1, 4]).split(2)


        input_tokens_enc = self.input_tokens.unsqueeze(0).expand(old_bsz, -1).to(self.device)
        temp_control_enc = self.wte_enc(input_tokens_enc)
        past_key_values_enc = self.control_trans_enc(temp_control_enc)  # bsz, seqlen, layer*emb
        bsz_enc, seqlen, _ = past_key_values_enc.shape
        past_key_values_enc = past_key_values_enc.view(bsz_enc, seqlen, self.match_n_layer * 2, self.match_n_head,
                                                 self.match_n_embd)
        past_key_values_enc = self.dropout(past_key_values_enc)
        past_key_values_enc = past_key_values_enc.permute([2, 0, 3, 1, 4]).split(2)

        result = []
        for i, key_val in enumerate(past_key_values):
            temp_dict = {'self': {"prev_key": key_val[0].contiguous(),
                                  "prev_value": key_val[1].contiguous(),
                                  "prev_key_padding_mask": torch.zeros(bsz, seqlen).to(key_val.device).bool() #bsz, preseqlen
                                 },
                        }

            key_val2 = past_key_values2[i]
            temp_dict['encoder_decoder'] = {"prev_key": key_val2[0].contiguous(),
                                            "prev_value": key_val2[1].contiguous(),
                                            "prev_key_padding_mask": torch.zeros(bsz, seqlen).to(key_val2.device).bool()
                                            }

            key_val_enc = past_key_values_enc[i]
            temp_dict['encoder'] = {"prev_key": key_val_enc[0].contiguous(),
                                    "prev_value": key_val_enc[1].contiguous(),
                                    "prev_key_padding_mask": torch.zeros(bsz_enc, seqlen).to(key_val_enc.device).bool()
                                    }
            result.append(temp_dict)

        return result

    def forward(self, input_ids=None, attention_mask=None, decoder_input_ids=None, decoder_attention_mask=None, past_key_values = None, use_cache=False, use_prefix=True, output_hidden_states=True, output_attentions=True, return_dict=True,**model_kwargs):

        bsz = input_ids.shape[0]
        past_key_values_prompt = self.get_prompt(bsz=bsz)
        if past_key_values is not None:
            assert False, "Attention, use past_key_values for other things"
        else:
            past_key_values = past_key_values_prompt
        output = self.bart(
            input_ids=input_ids,
            attention_mask=attention_mask,
            decoder_input_ids=decoder_input_ids,
            decoder_attention_mask=decoder_attention_mask,
            past_key_values=past_key_values,
            use_cache=use_cache,
            use_prefix=use_prefix,
            output_hidden_states=output_hidden_states,
            output_attentions=output_attentions,
            return_dict=return_dict,
            **model_kwargs
        )

        decoder_feature = self.w_de(output.decoder_hidden_states[-1])
        p_gen = torch.sigmoid(decoder_feature)
        
        copy_dist = torch.zeros(output.logits.shape,device=self.device)
        att = output.cross_attentions[-1][:,:,:,self.preseqlen:].mean(1).detach()
        input_ids=input_ids.long()
        for i in range(input_ids.shape[0]):
            input_ids_sorted, _ = input_ids[i].sort()
            total_sorted, _ = (att[i,:,:]+input_ids[i]).sort()
            copy_dist[i,:,input_ids_sorted] = total_sorted%1
        copy_dist = copy_dist/(copy_dist.sum(-1).unsqueeze(-1))
        output = (1-p_gen)*copy_dist + (p_gen)*torch.nn.Softmax(dim=-1)(output.logits)
        return output

    def generative_step(self,input_ids=None, max_length=None, min_length=None, num_beams=None, length_penalty=None, no_repeat_ngram_size=None, early_stopping=None,**model_kwargs):
        # TODO(LISA)
        bsz = input_ids.shape[0]
        past_key_values = self.get_prompt(bsz=bsz,sample_size=num_beams)
        model_kwargs["copy"]=True
        model_kwargs["w_de"]=self.w_de
        model_kwargs["preseqlen"]=self.preseqlen
        model_kwargs["encoder_input_ids"]=input_ids
        
        generated_ids = self.bart.generate(
            input_ids=input_ids,
            past_key_values=past_key_values,
            use_cache=True,
            length_penalty=length_penalty,
            use_prefix=True,
            num_beams=num_beams,
            min_length=min_length,
            max_length=max_length,
            no_repeat_ngram_size=no_repeat_ngram_size,
            early_stopping=early_stopping,
            **model_kwargs
        )

        return generated_ids
