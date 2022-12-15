# -*- coding:utf-8 -*-

import json, os
from tqdm import tqdm

import torch
from torch.utils.data import RandomSampler, SequentialSampler, Dataset
from pytorch_lightning import LightningDataModule 
from transformers import AutoTokenizer
import dgl
from dgl.dataloading import GraphDataLoader

class QSGDataset(Dataset):
    def __init__(self, cfg, features, split):
        super().__init__()
        self.features = features
        self.split = split
        self.cfg = cfg

    def __len__(self):
        return len(self.features)

    def __getitem__(self, i):
        if self.split == 'train' or self.split == 'valid':
            output = {
                'enc_ids' : torch.tensor(self.features[i]['enc_ids'], dtype = torch.long),
                'enc_mask' : torch.tensor(self.features[i]['enc_mask'], dtype = torch.long),
                'token2node' : torch.tensor(self.features[i]['token2node'], dtype = torch.long),
                'dec_ids' : torch.tensor(self.features[i]['dec_ids'], dtype = torch.long),
                'dec_mask' : torch.tensor(self.features[i]['dec_mask'], dtype = torch.long),
                'label_ids' : torch.tensor(self.features[i]['label_ids'], dtype = torch.long),
                'summary' : self.features[i]['summary']
            }

        elif self.split == 'test':
            output = {
                'enc_ids' : torch.tensor(self.features[i]['enc_ids'], dtype = torch.long),
                'enc_mask' : torch.tensor(self.features[i]['enc_mask'], dtype = torch.long),
                'token2node' : torch.tensor(self.features[i]['token2node'], dtype = torch.long),
                'summary' : self.features[i]['summary']
            }

        src = [e[0] for e in self.features[i]['edge']]
        dst = [e[1] for e in self.features[i]['edge']]
        g = dgl.graph((src, dst))
        g = dgl.to_simple(g)
        g = dgl.to_bidirected(g)
        g.ndata['pr_score'] = torch.tensor(self.features[i]['pr_score'], dtype = torch.float)

        aux_label = self.features[i]['aux_label']
        if len(aux_label) < self.cfg.enc_len:
            aux_label = aux_label + [-100] * (self.cfg.enc_len - len(aux_label))
        else:
            aux_label = aux_label[:self.cfg.enc_len]
        output['aux_label'] = torch.tensor(aux_label, dtype = torch.long)

        return output, g

class DataModule(LightningDataModule):
    def __init__(self, cfg):
        super().__init__()
        self.cfg = cfg
        self.tokenizer = AutoTokenizer.from_pretrained(cfg.model_name)

    def prepare_data(self):
        AutoTokenizer.from_pretrained(self.cfg.model_name)

    def _read_data(self):
        data = {}
        splits = ['train', 'valid', 'test']
        for split in splits:
            data_filename = os.path.join(self.cfg.data_dir, split + '.json')
            with open(data_filename, 'r') as f:
                data[split] = json.load(f)
        
        return data

    def setup(self, stage = None):
        data = self._read_data()

        self.dataset = {}
        for split in data:
            self.dataset[split] = QSGDataset(
                self.cfg,
                self._to_features(data[split]),
                split
            )

    def train_dataloader(self):
        sampler = RandomSampler(self.dataset['train'])
        return GraphDataLoader(self.dataset['train'], sampler = sampler, batch_size = self.cfg.batch_size)

    def val_dataloader(self):
        '''
            mode : {
                'check_loss',
                'check_rouge'
            }
        '''
        sampler = SequentialSampler(self.dataset['valid'])

        return GraphDataLoader(self.dataset['valid'], sampler = sampler, batch_size = self.cfg.batch_size)


    def test_dataloader(self):
        sampler = SequentialSampler(self.dataset['test'])
        return GraphDataLoader(self.dataset['test'], sampler = sampler, batch_size = self.cfg.batch_size)

    def _to_features(self, data_split):
        output = []

        for dp in tqdm(data_split):
            output.append(self._mk_feature(dp))

        return output

    def _mk_feature(self, dp):
        bos_token_id = self.tokenizer.bos_token_id # 0
        cls_token_id = self.tokenizer.cls_token_id # 0
        sep_token_id = self.tokenizer.sep_token_id # 2
        pad_token_id = self.tokenizer.pad_token_id # 1
        eos_token_id = self.tokenizer.eos_token_id # 2
        ignore_token_id = -100

        token_list = []
        token2node = []
        edge = dp['edge']
        pr_score = dp['pr_score']
        aux_label = dp['aux_label']

        for word, word2node_idx in zip(dp['word'], dp['word2node']):
            tokens = self.tokenizer.tokenize(" " + word)

            token_list += tokens
            token2node += [word2node_idx] * len(tokens)

        enc_ids = [cls_token_id] + self.tokenizer.convert_tokens_to_ids(token_list) + [sep_token_id]
        enc_mask = [1] * len(enc_ids)
        token2node = [-1] + token2node + [-1]
        if len(enc_ids) <= self.cfg.enc_len:
            padding_size = self.cfg.enc_len - len(enc_ids)
            enc_ids += [pad_token_id] * padding_size
            enc_mask += [0] * padding_size
            token2node += [-1] * padding_size
        else:
            enc_ids = enc_ids[:self.cfg.enc_len]
            enc_mask = enc_mask[:self.cfg.enc_len]
            token2node = token2node[:self.cfg.enc_len]

        summary_token = self.tokenizer.tokenize(dp['summary'])
        temp_ids = self.tokenizer.convert_tokens_to_ids(summary_token)
        dec_mask = [1] * (len(temp_ids) + 1)
        if len(temp_ids) + 1 <= self.cfg.dec_len:
            padding_size = self.cfg.dec_len - len(temp_ids) - 1
            dec_ids = [bos_token_id] + temp_ids + [pad_token_id] * padding_size
            dec_mask += [0] * padding_size
            label_ids = temp_ids + [eos_token_id] + [ignore_token_id] * padding_size
        else:
            temp_ids = temp_ids[:self.cfg.dec_len - 1]
            dec_ids = [bos_token_id] + temp_ids
            dec_mask = dec_mask[:self.cfg.dec_len]
            label_ids = temp_ids + [eos_token_id]

        assert len(enc_ids) == self.cfg.enc_len
        assert len(enc_mask) == self.cfg.enc_len
        assert len(token2node) == self.cfg.enc_len
        assert len(dec_ids) == self.cfg.dec_len
        assert len(dec_mask) == self.cfg.dec_len
        assert len(label_ids) == self.cfg.dec_len

        output = {
            'enc_ids' : enc_ids,
            'enc_mask' : enc_mask,
            'token2node' : token2node,
            'edge' : edge,
            'pr_score' : pr_score,
            'aux_label' : aux_label,
            'dec_ids' : dec_ids,
            'dec_mask' : dec_mask,
            'label_ids' : label_ids,
            'summary' : dp['summary']
        }

        return output