# -*- coding: utf-8 -*-


import numpy as np
import pandas as pd
import os
import sys
import gzip
import json
import random
import torch
import csv
import json
import datasets
import re

from dataclasses import dataclass
from pathlib import Path
from torch.utils.data import TensorDataset
from transformers import AutoTokenizer
from tqdm import tqdm
from typing import List, Dict, Tuple, Optional

support_set_ratio = 0.5
num_of_validation = 1000
tokenizer = AutoTokenizer.from_pretrained("facebook/bart-large")

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

def main():
    data_set_list = ["aeslc","billsum","gigaword","multi_news","reddit_tifu","arxiv","pubmed","wikihow","bigpatent","xsum","cnn_dailymail","newsroom"]
    split_type_list = ["train","validation","test"]
    for data_set in data_set_list:
        for split_type in split_type_list:
            global corpus_path
            corpus_path = "corpus"
            enc_max_length = 1024
            dec_max_length = 512
            if Path(f"./{corpus_path}/{data_set}_{split_type}_support").exists() or Path(f"./{corpus_path}/{data_set}_{split_type}_query").exists() or Path(f"./{corpus_path}/{data_set}_{split_type}").exists() or Path(f"./{corpus_path}/{data_set}_meta_{split_type}").exists():
                continue
            if data_set == "aeslc":
                make_tokenized_data_set(split_type,data_set,"email_body","subject_line",enc_max_length,dec_max_length)
            elif data_set == "billsum":
                make_tokenized_data_set(split_type,data_set,"text","summary",enc_max_length,dec_max_length)
            elif data_set == "gigaword":
                make_tokenized_data_set(split_type,data_set,"document","summary",enc_max_length,dec_max_length)
            elif data_set == "multi_news":
                make_tokenized_data_set(split_type,data_set,"document","summary",enc_max_length,dec_max_length)
            elif data_set == "newsroom":
                make_tokenized_data_set_newsroom(split_type,data_set,"text","summary",enc_max_length,dec_max_length)
            elif data_set == "reddit_tifu":
                make_tokenized_data_set_reddit_tifu(data_set,"documents","tldr",enc_max_length,dec_max_length)
            elif data_set == "arxiv":
                make_tokenized_data_set_pubmed_arxiv(split_type,data_set,enc_max_length,dec_max_length)
            elif data_set == "pubmed":
                make_tokenized_data_set_pubmed_arxiv(split_type,data_set,enc_max_length,dec_max_length)
            elif data_set == "wikihow":
                make_tokenized_data_set(split_type,data_set,"article","summary",enc_max_length,dec_max_length) #157306/1000/5579
            elif data_set == "bigpatent":
                make_tokenized_data_set_bigpatent(split_type,data_set,"./release/bigPatentData",enc_max_length,dec_max_length)
            elif data_set == "xsum":
                make_tokenized_data_set(split_type,data_set,"document","summary",enc_max_length,dec_max_length)
            elif data_set == "cnn_dailymail":
                make_tokenized_data_set_cnn_dailymail(split_type,data_set,"article","highlights",enc_max_length,dec_max_length)
            elif data_set == "newsroom":
                make_tokenized_data_set_newsroom(split_type,data_set,"text","summary",enc_max_length,dec_max_length)

def make_tokenized_data_set(split_type,data_set,text_name,summary_name,enc_max_length,dec_max_length):
    if data_set == "billsum" and split_type == "validation":
        datas = datasets.load_dataset(f"{data_set}", split="ca_test")
        datas_set = make_data_set(datas, text_name, summary_name, split_type)
    elif data_set == "wikihow":
        with open(f"./release/wikihow_{split_type}.json", 'r') as outfile:
            datas = json.load(outfile)
        datas_set = make_wikihow_data_set(datas, text_name, summary_name, split_type)
    else:
        datas = datasets.load_dataset(f"{data_set}", split=f"{split_type}")
        datas_set = make_data_set(datas, text_name, summary_name, split_type)

    if split_type == "train":
        num_of_support_examples = int(len(datas_set) * support_set_ratio)
        datas_train_set_support = datas_set[:num_of_support_examples]
        features_train_support = make_features_to_examples(datas_train_set_support,split_type,enc_max_length,dec_max_length)
        save_features_file = Path(f"./{corpus_path}/{data_set}_{split_type}_support")
        torch.save(features_train_support, save_features_file)

        datas_train_set_query = datas_set[num_of_support_examples:]
        features_train_query = make_features_to_examples(datas_train_set_query,split_type,enc_max_length,dec_max_length)
        save_features_file = Path(f"./{corpus_path}/{data_set}_{split_type}_query")
        torch.save(features_train_query, save_features_file)

    elif split_type == "validation":
        num_of_meta_validation = int(len(datas_set)/2)
        datas_meta_validation = datas_set[:num_of_meta_validation]
        datas_validation = datas_set[num_of_meta_validation:]
        features_meta_validation = make_features_to_examples(datas_meta_validation,"meta_"+split_type,enc_max_length,dec_max_length)
        features_validation = make_features_to_examples(datas_validation,split_type,enc_max_length,dec_max_length)
        torch.save(features_meta_validation,Path(f"./{corpus_path}/{data_set}_meta_validation"))
        torch.save(features_validation, Path(f"./{corpus_path}/{data_set}_{split_type}"))

    elif split_type == "test":
        features_test, labels = make_features_to_examples(datas_set,split_type,enc_max_length,dec_max_length)
        save_features_file = Path(f"./{corpus_path}/{data_set}_{split_type}")
        torch.save((features_test, labels), save_features_file)

def make_tokenized_data_set_newsroom(split_type,data_set,text_name,summary_name,enc_max_length,dec_max_length):
    with open(f'./release/newsroom_{split_type}.jsonl', 'r') as json_file:
        datas = list(json_file)
    datas_set = make_newsroom_data_set(datas, text_name, summary_name, split_type)

    if split_type == "train":
        num_of_support_examples = int(len(datas_set) * support_set_ratio)
        datas_train_set_support = datas_set[:num_of_support_examples]
        features_train_support = make_features_to_examples(datas_train_set_support,split_type,enc_max_length,dec_max_length)
        save_features_file = Path(f"./{corpus_path}/{data_set}_{split_type}_support")
        torch.save(features_train_support, save_features_file)

        datas_train_set_query = datas_set[num_of_support_examples:]
        features_train_query = make_features_to_examples(datas_train_set_query,split_type,enc_max_length,dec_max_length)
        save_features_file = Path(f"./{corpus_path}/{data_set}_{split_type}_query")
        torch.save(features_train_query, save_features_file)

    elif split_type == "validation":
        num_of_meta_validation = int(len(datas_set)/2)
        datas_meta_validation = datas_set[:num_of_meta_validation]
        datas_validation = datas_set[num_of_meta_validation:]
        features_meta_validation = make_features_to_examples(datas_meta_validation,"meta_"+split_type,enc_max_length,dec_max_length)
        features_validation = make_features_to_examples(datas_validation,split_type,enc_max_length,dec_max_length)
        torch.save(features_meta_validation,Path(f"./{corpus_path}/{data_set}_meta_validation"))
        torch.save(features_validation, Path(f"./{corpus_path}/{data_set}_{split_type}"))

    elif split_type == "test":
        features_test, labels = make_features_to_examples(datas_set,split_type,enc_max_length,dec_max_length)
        save_features_file = Path(f"./{corpus_path}/{data_set}_{split_type}")
        torch.save((features_test, labels), save_features_file)

def make_tokenized_data_set_reddit_tifu(data_set,text_name,summary_name,enc_max_length,dec_max_length):
    datas = datasets.load_dataset("reddit_tifu", 'long')["train"]
    data_train_set, data_validation_set, data_test_set = make_reddit_tifu_data_split_set(datas, text_name, summary_name)

    num_of_support_examples = int(len(data_train_set) * support_set_ratio)
    datas_train_set_support = data_train_set[:num_of_support_examples]
    features_train_support = make_features_to_examples(datas_train_set_support,"train",enc_max_length,dec_max_length)
    save_features_file = Path(f"./{corpus_path}/{data_set}_train_support")
    torch.save(features_train_support, save_features_file)

    datas_train_set_query = data_train_set[num_of_support_examples:]
    features_train_query = make_features_to_examples(datas_train_set_query,"train",enc_max_length,dec_max_length)
    save_features_file = Path(f"./{corpus_path}/{data_set}_train_query")
    torch.save(features_train_query, save_features_file)

    num_of_meta_validation = int(len(data_validation_set) / 2)
    datas_meta_validation = data_validation_set[:num_of_meta_validation]
    datas_validation = data_validation_set[num_of_meta_validation:]
    features_meta_validation = make_features_to_examples(datas_meta_validation, "meta_validation",enc_max_length,dec_max_length)
    features_validation = make_features_to_examples(datas_validation, "validation",enc_max_length,dec_max_length)
    torch.save(features_meta_validation, Path(f"./{corpus_path}/{data_set}_meta_validation"))
    torch.save(features_validation, Path(f"./{corpus_path}/{data_set}_validation"))

    features_test, labels_test = make_features_to_examples(data_test_set,"test",enc_max_length,dec_max_length)
    save_features_file = Path(f"./{corpus_path}/{data_set}_test")
    torch.save((features_test, labels_test), save_features_file)

def make_tokenized_data_set_pubmed_arxiv(split_type,data_set,enc_max_length,dec_max_length):
    datas_set = make_pubmed_arxiv_data_set(f"./release/{data_set}_{split_type}.txt",split_type)

    if split_type == "train":
        num_of_support_examples = int(len(datas_set) * support_set_ratio)
        datas_train_set_support = datas_set[:num_of_support_examples]
        features_train_support = make_features_to_examples(datas_train_set_support,split_type,enc_max_length,dec_max_length)
        save_features_file = Path(f"./{corpus_path}/{data_set}_{split_type}_support")
        torch.save(features_train_support, save_features_file)

        datas_train_set_query = datas_set[num_of_support_examples:]
        features_train_query = make_features_to_examples(datas_train_set_query,split_type,enc_max_length,dec_max_length)
        save_features_file = Path(f"./{corpus_path}/{data_set}_{split_type}_query")
        torch.save(features_train_query, save_features_file)

    elif split_type == "validation":
        num_of_meta_validation = int(len(datas_set)/2)
        datas_meta_validation = datas_set[:num_of_meta_validation]
        datas_validation = datas_set[num_of_meta_validation:]
        features_meta_validation  = make_features_to_examples(datas_meta_validation,"meta_"+split_type,enc_max_length,dec_max_length)
        features_validation  = make_features_to_examples(datas_validation,split_type,enc_max_length,dec_max_length)
        torch.save(features_meta_validation,Path(f"./{corpus_path}/{data_set}_meta_validation"))
        torch.save(features_validation, Path(f"./{corpus_path}/{data_set}_{split_type}"))

    elif split_type == "test":
        features_test, labels = make_features_to_examples(datas_set,split_type,enc_max_length,dec_max_length)
        save_features_file = Path(f"./{corpus_path}/{data_set}_{split_type}")
        torch.save((features_test, labels), save_features_file)

def make_tokenized_data_set_bigpatent(split_type,data_set,data_dir,enc_max_length,dec_max_length):
    datas_set = make_bigpatent_train_data_set(data_dir,split_type)

    if split_type == "train":
        num_of_support_examples = int(len(datas_set) * support_set_ratio)
        datas_train_set_support = datas_set[:num_of_support_examples]
        features_train_support = make_features_to_examples(datas_train_set_support,split_type,enc_max_length,dec_max_length)
        save_features_file = Path(f"./{corpus_path}/{data_set}_{split_type}_support")
        torch.save(features_train_support, save_features_file)

        datas_train_set_query = datas_set[num_of_support_examples:]
        features_train_query = make_features_to_examples(datas_train_set_query,split_type,enc_max_length,dec_max_length)
        save_features_file = Path(f"./{corpus_path}/{data_set}_{split_type}_query")
        torch.save(features_train_query, save_features_file)

    elif split_type == "validation":
        num_of_meta_validation = int(len(datas_set)/2)
        datas_meta_validation = datas_set[:num_of_meta_validation]
        datas_validation = datas_set[num_of_meta_validation:]
        features_meta_validation = make_features_to_examples(datas_meta_validation,"meta_"+split_type,enc_max_length,dec_max_length)
        features_validation = make_features_to_examples(datas_validation,split_type,enc_max_length,dec_max_length)
        torch.save(features_meta_validation,Path(f"./{corpus_path}/{data_set}_meta_validation"))
        torch.save(features_validation, Path(f"./{corpus_path}/{data_set}_{split_type}"))

    elif split_type == "test":
        features_test, labels = make_features_to_examples(datas_set,split_type,enc_max_length,dec_max_length)
        save_features_file = Path(f"./{corpus_path}/{data_set}_{split_type}")
        torch.save((features_test, labels), save_features_file)

def make_tokenized_data_set_cnn_dailymail(split_type,data_set,text_name,summary_name,enc_max_length,dec_max_length):
    datas = datasets.load_dataset(f"{data_set}","3.0.0", split=f"{split_type}")
    datas_set = make_data_set_cnn_dailymail(datas, text_name, summary_name, split_type)

    if split_type == "train":
        num_of_support_examples = int(len(datas_set) * support_set_ratio)
        datas_train_set_support = datas_set[:num_of_support_examples]
        features_train_support = make_features_to_examples(datas_train_set_support,split_type,enc_max_length,dec_max_length)
        save_features_file = Path(f"./{corpus_path}/{data_set}_{split_type}_support")
        torch.save(features_train_support, save_features_file)

        datas_train_set_query = datas_set[num_of_support_examples:]
        features_train_query = make_features_to_examples(datas_train_set_query,split_type,enc_max_length,dec_max_length)
        save_features_file = Path(f"./{corpus_path}/{data_set}_{split_type}_query")
        torch.save(features_train_query, save_features_file)

    elif split_type == "validation":
        num_of_meta_validation = int(len(datas_set) / 2)
        datas_meta_validation = datas_set[:num_of_meta_validation]
        datas_validation = datas_set[num_of_meta_validation:]
        features_meta_validation  = make_features_to_examples(datas_meta_validation, "meta_"+split_type,enc_max_length,dec_max_length)
        features_validation = make_features_to_examples(datas_validation, split_type,enc_max_length,dec_max_length)
        torch.save(features_meta_validation, Path(f"./{corpus_path}/{data_set}_meta_validation"))
        torch.save(features_validation, Path(f"./{corpus_path}/{data_set}_{split_type}"))

    elif split_type == "test":
        features_test, labels = make_features_to_examples(datas_set,split_type,enc_max_length,dec_max_length)
        save_features_file = Path(f"./{corpus_path}/{data_set}_{split_type}")
        torch.save((features_test, labels), save_features_file)

def preprocessing_string(data_string):
    data_string = data_string.replace(" .", ". ")
    data_string = re.sub(' +', ' ', data_string)
    data_string = data_string.rstrip().lstrip()
    return data_string

def make_data_set(data_dict,text_name,summary_name,split_type):
    data_set = []
    if split_type == "train":
        if len(data_dict[text_name]) >40000:
            idx_list = random.sample(list(range(len(data_dict[text_name]))), 40000)
        else:
            idx_list = random.sample(list(range(len(data_dict[text_name]))),len(data_dict[text_name]))
    elif split_type == "validation":
        if len(data_dict[text_name]) >1000:
            idx_list = random.sample(list(range(len(data_dict[text_name]))), num_of_validation)
        else:
            idx_list = random.sample(list(range(len(data_dict[text_name]))),len(data_dict[text_name]))
    elif split_type == "test":
        idx_list = list(range(len(data_dict[text_name])))
    for idx in tqdm(idx_list):
        data_dict_idx = data_dict[idx]
        document = preprocessing_string(data_dict_idx[text_name])
        data_set.append((document,data_dict_idx[summary_name]))
    return data_set

def make_newsroom_data_set(data_dict_list,text_name,summary_name,split_type):
    data_set = []
    if split_type == "train":
        if len(data_dict_list) >40000:
            idx_list = random.sample(list(range(len(data_dict_list))),40000)
        else:
            idx_list = random.sample(list(range(len(data_dict_list))),len(data_dict_list))
    elif split_type == "validation":
        idx_list = random.sample(list(range(len(data_dict_list))), num_of_validation)
    elif split_type == "test":
        idx_list = list(range(len(data_dict_list)))
    for idx in tqdm(idx_list):
        data_dict_idx = json.loads(data_dict_list[idx])
        document = preprocessing_string(data_dict_idx[text_name])
        if len(data_dict_idx[summary_name])<20:
            continue
        data_set.append((document,data_dict_idx[summary_name]))
    return data_set

def make_reddit_tifu_data_split_set(data_dict,text_name,summary_name):
    idx_list  = list(range(len(data_dict)))
    random.shuffle(idx_list)
    num_of_train_dict = 38122
    num_of_validation_dict = num_of_validation
    num_of_test_dict = 2017
    data_train_set = []
    data_validation_set = []
    data_test_set = []
    for idx in idx_list[:num_of_train_dict]:
        data_dict_idx = data_dict[idx]
        document = preprocessing_string(data_dict_idx[text_name])
        data_train_set.append((document,data_dict_idx[summary_name]))
    for idx in idx_list[num_of_train_dict:num_of_train_dict+num_of_validation_dict]:
        data_dict_idx = data_dict[idx]
        document = preprocessing_string(data_dict_idx[text_name])
        data_validation_set.append((document,data_dict_idx[summary_name]))
    for idx in idx_list[-num_of_test_dict:]:
        data_dict_idx = data_dict[idx]
        document = preprocessing_string(data_dict_idx[text_name])
        data_test_set.append((document,data_dict_idx[summary_name]))
    return data_train_set, data_validation_set, data_test_set

def make_pubmed_arxiv_data_set(data_dir,split_type):
    data_set = []
    with open(data_dir, "r", encoding='UTF-8') as data_txt:
        data_list = list(data_txt)
    if split_type == "train":
        if len(data_list) > 40000:
            idx_list = random.sample(list(range(len(data_list))),40000)
        else:
            idx_list = random.sample(list(range(len(data_list))),len(data_list))
    elif split_type == "validation":
        idx_list = random.sample(list(range(len(data_list))), num_of_validation)
    elif split_type == "test":
        idx_list = list(range(len(data_list)))
    for idx in tqdm(idx_list):
        data_dict = json.loads(data_list[idx][:-1])
        document = ""
        for sentence in data_dict["article_text"]:
            document = document + sentence
        summary = ""
        for sentence in data_dict["abstract_text"]:
            summary = summary + sentence[4:-5]
        document = preprocessing_string(document)
        if document =="" or summary=="":
            continue
        data_set.append((document,summary))
    return data_set

def make_wikihow_data_set(data_dict,text_name,summary_name,split_type):
    data_set = []
    if split_type == "train":
        if len(data_dict[text_name]) >40000:
            idx_list = random.sample(list(range(len(data_dict[text_name]))), 40000)
        else:
            idx_list = random.sample(list(range(len(data_dict[text_name]))),len(data_dict[text_name]))
    elif split_type == "validation":
        idx_list = random.sample(list(range(len(data_dict[text_name]))), num_of_validation)
    elif split_type == "test":
        idx_list = list(range(len(data_dict[text_name])))
    for idx in tqdm(idx_list):
        document = preprocessing_string(data_dict[text_name][idx])
        data_set.append((document,data_dict[summary_name][idx]))
    return data_set

def make_bigpatent_train_data_set(data_dir,split_type):
    cpc_code_list = ["a","b","c","d","e","f","g","h","y"]
    data_set = []
    if split_type == "train":
        num_of_train_data = 1207222
        count = 0
        idx_list = random.sample(list(range(num_of_train_data)),40000)
    if split_type == "validation":
        num_of_validation_data = 67068
        count = 0
        idx_list = random.sample(list(range(num_of_validation_data)),num_of_validation)
    for cpc_code in tqdm(cpc_code_list):
        file_names = os.listdir(os.path.join(data_dir,split_type,cpc_code))
        for file_name in file_names:
            with gzip.open(os.path.join(data_dir,split_type,cpc_code,file_name),'r') as data_txt:
                for data in data_txt:
                    if split_type == "train":
                        if count in idx_list:
                            data_json = json.loads(data[:-1])
                            document = preprocessing_string(data_json["description"])
                            data_set.append((document,data_json["abstract"]))
                        count +=1
                    elif split_type == "validation":
                        if count in idx_list:
                            data_json = json.loads(data[:-1])
                            document = preprocessing_string(data_json["description"])
                            data_set.append((document,data_json["abstract"]))
                        count +=1
                    elif split_type == "test":
                        data_json = json.loads(data[:-1])
                        document = preprocessing_string(data_json["description"])
                        data_set.append((document, data_json["abstract"]))
    return data_set

def make_data_set_cnn_dailymail(data_dict,text_name,summary_name,split_type):
    data_set = []
    if split_type == "train":
        if len(data_dict[text_name]) >40000:
            idx_list = random.sample(list(range(len(data_dict[text_name]))), 40000)
        else:
            idx_list = random.sample(list(range(len(data_dict[text_name]))),len(data_dict[text_name]))
    elif split_type == "validation":
        idx_list = random.sample(list(range(len(data_dict[text_name]))), num_of_validation)
    elif split_type == "test":
        idx_list = list(range(len(data_dict[text_name])))
    for idx in tqdm(idx_list):
        data_dict_idx = data_dict[idx]
        document = data_dict_idx[text_name]
        document_idx = 0
        if document[:50].find("(CNN) -- ") != -1:
            document_idx = document[:50].find("(CNN) -- ") + 9
        elif document[:50].find("(CNN)") != -1:
            document_idx = document[:50].find("(CNN)") + 5
        document = document[document_idx:]
        document = preprocessing_string(document)
        data_set.append((document,data_dict_idx[summary_name]))
    return data_set

def make_examples_from_data_set(data_set):
    examples = list()
    for data in data_set:
        words = data[0]
        labels = data[1]
        examples.append(InputExample(words=words, labels=labels))
    return examples

def make_features_to_examples(examples,split_type,enc_max_length,dec_max_length):
    features = []
    labels = []
    for example in examples:
        word = example[0]
        label = example[1]

        enc_tokens = tokenizer(word, padding="max_length",truncation=True, max_length=enc_max_length)
        encoder_ids = enc_tokens.input_ids
        encoder_mask = enc_tokens.attention_mask

        assert len(encoder_ids) == enc_max_length

        if split_type == "train" or split_type == 'meta_validation':
            dec_tokens = tokenizer(label, padding="max_length",truncation=True, max_length=dec_max_length)
            decoder_ids = [2]+ dec_tokens.input_ids[:-1]
            decoder_mask = dec_tokens.attention_mask
            label_ids = decoder_ids[1:] + [-100]
            label_ids = [-100 if token == tokenizer.pad_token_id else token for token in label_ids]

            assert len(encoder_mask) == enc_max_length
            assert len(decoder_ids) == dec_max_length
            assert len(decoder_mask) == dec_max_length
            assert len(label_ids) == dec_max_length

            features.append(
                InputFeatures(
                    encoder_ids=encoder_ids,
                    encoder_mask=encoder_mask,
                    decoder_ids=decoder_ids,
                    decoder_mask=decoder_mask,
                    label_ids=label_ids,
                )
            )

        elif split_type == "test" or split_type == 'validation':
            labels.append(label)
            features.append(
                InputFeatures(
                    encoder_ids=encoder_ids,
                    encoder_mask=encoder_mask,
                    decoder_ids=None,
                    decoder_mask=None,
                    label_ids=None,
                )
            )
    if split_type == "train" or  split_type == 'meta_validation':
        return features
    elif split_type == "test" or split_type == 'validation':
        return features, labels

if __name__ == "__main__":
    main()