# -*- coding: utf-8 -*-

import torch
import torch.nn as nn
from pytorch_lightning import LightningModule
from torch.optim import AdamW
from transformers import get_cosine_schedule_with_warmup
from torchmetrics.text.rouge import ROUGEScore

from src.models.qsg_transformer import QSGTransformerForConditionalGeneration

class Learner(LightningModule):
    def __init__(self, cfg):
        super().__init__()

        if cfg.exp_name == 'QSG-Transformer':
            self.model = QSGTransformerForConditionalGeneration(cfg)
        # elif cfg.exp_name == 'QSG-BART':
        #     self.model = QSGBartForConditionalGeneration(cfg)
        self.cfg = cfg

    def training_step(self, batch, batch_idx):
        output = self.model(
            enc_ids = batch[0]['enc_ids'],
            enc_mask = batch[0]['enc_mask'],
            token2node = batch[0]['token2node'],
            g = batch[1],
            dec_ids = batch[0]['dec_ids'],
            dec_mask = batch[0]['dec_mask'],
            enc_hidden_states = None,
            graph_hidden_states = None,
            graph_mask = None,
            past_key_values = None,
            label_ids = batch[0]['label_ids'],
            aux_label = batch[0]['aux_label'],
        )
        loss = output['loss']

        self.log('train_loss', loss, prog_bar = True)

        return {
            'loss' : loss
        }

    def validation_step(self, batch, batch_idx):
        output = self.model(
            enc_ids = batch[0]['enc_ids'],
            enc_mask = batch[0]['enc_mask'],
            token2node = batch[0]['token2node'],
            g = batch[1],
            dec_ids = batch[0]['dec_ids'],
            dec_mask = batch[0]['dec_mask'],
            enc_hidden_states = None,
            graph_hidden_states = None,
            graph_mask = None,
            past_key_values = None,
            label_ids = batch[0]['label_ids'],
            aux_label = batch[0]['aux_label'],
        )
        loss = output['loss']

        self.log('val_loss', loss, prog_bar = True)

        output = self.model.inference(
            enc_ids = batch[0]['enc_ids'],
            enc_mask = batch[0]['enc_mask'],
            token2node = batch[0]['token2node'],
            g = batch[1]
        )

        return {
            'loss' : loss,
            'golden_summary' : batch[0]['summary'],
            'generated_summary' : output['generated_summary']
        }

    def validation_epoch_end(self, outputs):
        avg_val_loss = torch.stack([x['loss'] for x in outputs]).mean()

        rouge = ROUGEScore(rouge_keys = ('rouge1', 'rouge2', 'rougeL'), use_stemmer = True)
        golden_summary_list, generated_summary_list = [], []
        for s in outputs:
            golden_summary_list += s['golden_summary']
            generated_summary_list += s['generated_summary']
        rouge_output = rouge(golden_summary_list, generated_summary_list)
        val_rouge1, val_rouge2, val_rougeL = float(rouge_output['rouge1_fmeasure']), float(rouge_output['rouge2_fmeasure']), float(rouge_output['rougeL_fmeasure'])
        val_rouge1, val_rouge2, val_rougeL = round(val_rouge1 * 100, 2), round(val_rouge2 * 100, 2), round(val_rougeL * 100, 2)

        self.log('avg_val_loss', avg_val_loss, prog_bar = True)
        self.log('val_rouge1', val_rouge1, prog_bar = True)
        self.log('val_rouge2', val_rouge2, prog_bar = True)
        self.log('val_rougeL', val_rougeL, prog_bar = True)

    def test_step(self, batch, batch_idx):
        output = self.model.inference(
            enc_ids = batch[0]['enc_ids'],
            enc_mask = batch[0]['enc_mask'],
            token2node = batch[0]['token2node'],
            g = batch[1]
        )

        return {
            'golden_summary' : batch[0]['summary'],
            'generated_summary' : output['generated_summary']
        }

    def test_epoch_end(self, outputs):
        rouge = ROUGEScore(rouge_keys = ('rouge1', 'rouge2', 'rougeL'), use_stemmer = True)
        golden_summary_list, generated_summary_list = [], []
        for s in outputs:
            golden_summary_list += s['golden_summary']
            generated_summary_list += s['generated_summary']
        rouge_output = rouge(golden_summary_list, generated_summary_list)
        val_rouge1, val_rouge2, val_rougeL = float(rouge_output['rouge1_fmeasure']), float(rouge_output['rouge2_fmeasure']), float(rouge_output['rougeL_fmeasure'])
        val_rouge1, val_rouge2, val_rougeL = round(val_rouge1 * 100, 2), round(val_rouge2 * 100, 2), round(val_rougeL * 100, 2)

        self.log('val_rouge1', val_rouge1, prog_bar = True)
        self.log('val_rouge2', val_rouge2, prog_bar = True)
        self.log('val_rougeL', val_rougeL, prog_bar = True)

    def configure_optimizers(self):
        optimizer = AdamW(self.parameters(), lr = self.cfg.learning_rate)

        scheduler = get_cosine_schedule_with_warmup(
            optimizer,
            num_warmup_steps = int(self.cfg.num_training_steps * 0.1),
            num_training_steps = self.cfg.num_training_steps
        )

        return [optimizer], [scheduler]