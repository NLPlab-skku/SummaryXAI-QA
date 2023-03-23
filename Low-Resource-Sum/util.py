# -*- coding: utf-8 -*-


import argparse
import logging
import numpy as np
import pandas as pd
import os
import random
import torch
import csv
import json
import datasets
import gc

from dataclasses import dataclass
from pathlib import Path
from torch.utils.data import TensorDataset, DataLoader, RandomSampler, SequentialSampler
from torch import nn
from torch.nn import CrossEntropyLoss, NLLLoss
from tqdm import tqdm, trange
from typing import List, Dict, Tuple, Optional
from rouge_score import rouge_scorer, scoring
from copy import deepcopy

logger = logging.getLogger(__name__)

def set_seed(args):
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed(args.seed)
    torch.cuda.manual_seed_all(args.seed)  # if use multi-GPU
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    np.random.seed(args.seed)
    random.seed(args.seed)

@dataclass(frozen=True)
class InputExample:
    words: List[str]
    labels: Optional[List[str]]

@dataclass(frozen=True)
class InputFeatures:
    encoder_ids : List[int]
    encoder_mask : List[int]
    decoder_ids : Optional[List[int]]
    decoder_mask : Optional[List[int]]
    label_ids : Optional[List[int]]

def load_and_cache_examples(args, data_name, mode):
    # Load data features from cache or dataset file
    if mode == "fine_tuning_10" or mode == "fine_tuning_100" or mode == "fine_tuning_1000":
        cached_features_file = Path(f"{args.data_dir}/{data_name}_train_support")
    elif  mode == "meta_validation_train":
        cached_features_file = Path(f"{args.data_dir}/{data_name}_train_query")
    else:
        cached_features_file = Path(f"{args.data_dir}/{data_name}_{mode}")

    if cached_features_file.exists():
        logger.info("Loading features from cached file %s", cached_features_file)
        if mode == "train_support" or mode == "train_query" or mode == "meta_validation" or mode == "fine_tuning_10" or mode == "fine_tuning_100" or  mode == "meta_validation_train":
            features = torch.load(cached_features_file)
        elif mode == "test" or mode == "validation":
            (features, labels) = torch.load(cached_features_file)
    else:
        logger.info("There is no dataset file at %s", args.data_dir)

    # Convert to Tensors and build dataset
    if mode == "train_support" or mode == "train_query" or mode == "meta_validation":
        all_encoder_ids = torch.tensor([f.encoder_ids for f in features], dtype=torch.long)
        all_encoder_mask = torch.tensor([f.encoder_mask for f in features], dtype=torch.float)
        all_decoder_ids = torch.tensor([f.decoder_ids for f in features], dtype=torch.long)
        all_decoder_mask = torch.tensor([f.decoder_mask for f in features], dtype=torch.float)
        all_label_ids = torch.tensor([f.label_ids for f in features], dtype=torch.long)
        dataset = TensorDataset(all_encoder_ids, all_encoder_mask, all_decoder_ids, all_decoder_mask, all_label_ids)

        return dataset

    elif mode == "fine_tuning_10" or mode == "fine_tuning_100" or mode == "fine_tuning_1000":
        num_of_sampled_examples = int(mode[12:])
        sampled_feature = random.sample(features,num_of_sampled_examples)
        print(len(sampled_feature))
        all_encoder_ids = torch.tensor([f.encoder_ids for f in sampled_feature], dtype=torch.long)
        all_encoder_mask = torch.tensor([f.encoder_mask for f in sampled_feature], dtype=torch.float)
        all_decoder_ids = torch.tensor([f.decoder_ids for f in sampled_feature], dtype=torch.long)
        all_decoder_mask = torch.tensor([f.decoder_mask for f in sampled_feature], dtype=torch.float)
        all_label_ids = torch.tensor([f.label_ids for f in sampled_feature], dtype=torch.long)
        dataset = TensorDataset(all_encoder_ids, all_encoder_mask, all_decoder_ids, all_decoder_mask, all_label_ids)

        return dataset

    elif mode == "meta_validation_train":
        num_of_sampled_examples = 10
        sampled_feature = random.sample(features,num_of_sampled_examples)
        print(len(sampled_feature))
        all_encoder_ids = torch.tensor([f.encoder_ids for f in sampled_feature], dtype=torch.long)
        all_encoder_mask = torch.tensor([f.encoder_mask for f in sampled_feature], dtype=torch.float)
        all_decoder_ids = torch.tensor([f.decoder_ids for f in sampled_feature], dtype=torch.long)
        all_decoder_mask = torch.tensor([f.decoder_mask for f in sampled_feature], dtype=torch.float)
        all_label_ids = torch.tensor([f.label_ids for f in sampled_feature], dtype=torch.long)
        dataset = TensorDataset(all_encoder_ids, all_encoder_mask, all_decoder_ids, all_decoder_mask, all_label_ids)

        return dataset

    elif mode == "test" or mode == "validation":
        all_encoder_ids = torch.tensor([f.encoder_ids for f in features], dtype=torch.long)
        all_encoder_mask = torch.tensor([f.encoder_mask for f in features], dtype=torch.float)
        dataset = TensorDataset(all_encoder_ids,all_encoder_mask)

        return dataset, labels

def make_support_query_tensordataset(args,source_dataset_list):
    train_support0_dataset = load_and_cache_examples(args, source_dataset_list[0], mode="train_support")
    train_support1_dataset = load_and_cache_examples(args, source_dataset_list[1], mode="train_support")
    train_support2_dataset = load_and_cache_examples(args, source_dataset_list[2], mode="train_support")

    train_query0_dataset = load_and_cache_examples(args, source_dataset_list[0], mode="train_query")
    train_query1_dataset = load_and_cache_examples(args, source_dataset_list[1], mode="train_query")
    train_query2_dataset = load_and_cache_examples(args, source_dataset_list[2], mode="train_query")

    batch_tasks = [(train_support0_dataset,train_query0_dataset),(train_support1_dataset,train_query1_dataset),(train_support2_dataset,train_query2_dataset)]

    return batch_tasks



