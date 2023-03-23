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

num_of_validation = 500

def main():
    data_set_list = ["aeslc","billsum","gigaword","multi_news","newsroom","reddit_tifu","arxiv","pubmed","wikihow","bigpatent","xsum","cnn_dailymail"]
    split_type_list = ["train"]
    for data_set in data_set_list:
        for split_type in split_type_list:
            if Path(f"./corpus_500/{data_set}").exists():
                continue
            if data_set == "aeslc":
                make_tokenized_data_set(split_type,data_set,"email_body")
            elif data_set == "billsum":
                make_tokenized_data_set(split_type,data_set,"text")
            elif data_set == "gigaword":
                make_tokenized_data_set(split_type,data_set,"document")
            elif data_set == "multi_news":
                make_tokenized_data_set(split_type,data_set,"document")
            elif data_set == "newsroom":
                make_tokenized_data_set_newsroom(split_type,data_set,"text")
            elif data_set == "reddit_tifu":
                make_tokenized_data_set_reddit_tifu(data_set,"documents")
            elif data_set == "arxiv":
                make_tokenized_data_set_pubmed_arxiv(split_type,data_set)
            elif data_set == "pubmed":
                make_tokenized_data_set_pubmed_arxiv(split_type,data_set)
            elif data_set == "wikihow":
                make_tokenized_data_set(split_type,data_set,"article") #157306/1000/5579
            elif data_set == "bigpatent":
                make_tokenized_data_set_bigpatent(split_type,data_set,"./release/bigPatentData")
            elif data_set == "xsum":
                make_tokenized_data_set(split_type,data_set,"document")
            elif data_set == "cnn_dailymail":
                make_tokenized_data_set_cnn_dailymail(split_type,data_set,"article","highlights")

def make_tokenized_data_set(split_type,data_set,text_name):
    if data_set == "billsum" and split_type == "train":
        datas = datasets.load_dataset(f"{data_set}", split="train")
        datas_set = make_data_set(datas, text_name)
    elif data_set == "wikihow":
        with open(f"./release/wikihow_{split_type}.json", 'r') as outfile:
            datas = json.load(outfile)
        datas_set = make_wikihow_data_set(datas, text_name)
    else:
        datas = datasets.load_dataset(f"{data_set}", split=f"{split_type}")
        datas_set = make_data_set(datas, text_name)

    with open(f"./corpus_500/{data_set}.json", 'w') as outfile:
        json.dump({"text":datas_set}, outfile)

def make_tokenized_data_set_newsroom(split_type,data_set,text_name):
    with open(f'./release/newsroom_{split_type}.jsonl', 'r') as json_file:
        datas = list(json_file)
    datas_set = make_newsroom_data_set(datas, text_name)

    with open(f"./corpus_500/{data_set}.json", 'w') as outfile:
        json.dump({"text":datas_set}, outfile)

def make_tokenized_data_set_reddit_tifu(data_set,text_name):
    datas = datasets.load_dataset("reddit_tifu", 'long')["train"]
    data_validation_set = make_reddit_tifu_data_split_set(datas, text_name)

    with open(f"./corpus_500/{data_set}.json", 'w') as outfile:
        json.dump({"text":data_validation_set}, outfile)

def make_tokenized_data_set_pubmed_arxiv(split_type,data_set):
    datas_set = make_pubmed_arxiv_data_set(f"./release/{data_set}_{split_type}.txt")

    with open(f"./corpus_500/{data_set}.json", 'w') as outfile:
        json.dump({"text":datas_set}, outfile)

def make_tokenized_data_set_bigpatent(split_type,data_set,data_dir):
    datas_set = make_bigpatent_train_data_set(data_dir,split_type)

    with open(f"./corpus_500/{data_set}.json", 'w') as outfile:
        json.dump({"text":datas_set}, outfile)

def make_tokenized_data_set_cnn_dailymail(split_type,data_set,text_name,summary_name):
    datas = datasets.load_dataset(f"{data_set}","3.0.0", split=f"{split_type}")
    datas_set = make_data_set_cnn_dailymail(datas, text_name)

    with open(f"./corpus_500/{data_set}.json", 'w') as outfile:
        json.dump({"text":datas_set}, outfile)

def preprocessing_string(data_string):
    data_string = data_string.replace(" .", ". ")
    data_string = re.sub(' +', ' ', data_string)
    data_string = data_string.rstrip().lstrip()
    return data_string

def make_data_set(data_dict,text_name):
    data_set = []
    idx_list = random.sample(list(range(len(data_dict[text_name]))), num_of_validation)
    for idx in tqdm(idx_list):
        data_dict_idx = data_dict[idx]
        document = preprocessing_string(data_dict_idx[text_name])
        data_set.append(document)
    return data_set

def make_newsroom_data_set(data_dict_list,text_name):
    data_set = []
    idx_list = random.sample(list(range(len(data_dict_list))), num_of_validation)
    for idx in tqdm(idx_list):
        data_dict_idx = json.loads(data_dict_list[idx])
        document = preprocessing_string(data_dict_idx[text_name])
        data_set.append(document)
    return data_set

def make_reddit_tifu_data_split_set(data_dict,text_name):
    idx_list  = list(range(len(data_dict)))
    random.shuffle(idx_list)
    num_of_validation_dict = num_of_validation
    data_validation_set = []
    for idx in idx_list[:num_of_validation_dict]:
        data_dict_idx = data_dict[idx]
        document = preprocessing_string(data_dict_idx[text_name])
        data_validation_set.append(document)
    return data_validation_set

def make_pubmed_arxiv_data_set(data_dir):
    data_set = []
    with open(data_dir, "r", encoding='UTF-8') as data_txt:
        data_list = list(data_txt)
    idx_list = random.sample(list(range(len(data_list))), num_of_validation)
    for idx in tqdm(idx_list):
        data_dict = json.loads(data_list[idx][:-1])
        document = ""
        for sentence in data_dict["article_text"]:
            document = document + sentence
        document = preprocessing_string(document)
        if document =="":
            continue
        data_set.append(document)
    return data_set

def make_wikihow_data_set(data_dict,text_name):
    data_set = []
    idx_list = random.sample(list(range(len(data_dict[text_name]))), num_of_validation)
    for idx in tqdm(idx_list):
        document = preprocessing_string(data_dict[text_name][idx])
        data_set.append(document)
    return data_set

def make_bigpatent_train_data_set(data_dir,split_type):
    cpc_code_list = ["a","b","c","d","e","f","g","h","y"]
    data_set = []
    if split_type == "train":
        num_of_validation_data = 67068
        count = 0
        idx_list = random.sample(list(range(num_of_validation_data)),num_of_validation)
    for cpc_code in tqdm(cpc_code_list):
        file_names = os.listdir(os.path.join(data_dir,split_type,cpc_code))
        for file_name in file_names:
            with gzip.open(os.path.join(data_dir,split_type,cpc_code,file_name),'r') as data_txt:
                for data in data_txt:
                    if count in idx_list:
                        data_json = json.loads(data[:-1])
                        document = preprocessing_string(data_json["description"])
                        data_set.append(document)
                    count +=1
    return data_set

def make_data_set_cnn_dailymail(data_dict,text_name):
    data_set = []
    idx_list = random.sample(list(range(len(data_dict[text_name]))), num_of_validation)
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
        data_set.append(document)
    return data_set

if __name__ == "__main__":
    main()