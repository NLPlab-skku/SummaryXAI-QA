from transformers import AutoTokenizer
import json
from tqdm import tqdm
import math

def main():
    tokenizer = AutoTokenizer.from_pretrained("facebook/bart-large")
    data_set_list = ["aeslc","billsum","gigaword","multi_news","reddit_tifu","arxiv","pubmed","wikihow"]

    total_data_list = []
    for data_set in data_set_list:
        with open(f"./corpus_500/{data_set}.json", 'r') as json_file:
            data = json.load(json_file)
        total_data_list.append(data["text"])

    rank_length(total_data_list,tokenizer,data_set_list)
    rank_cosine(total_data_list,data_set_list)
    rank_precision(total_data_list,data_set_list)
    

def save_dict(rank_dict,file_name):
    with open(f"./corpus_500/{file_name}.json", 'w') as outfile:
        json.dump(rank_dict, outfile)
    
def rank_length(total_data_list,tokenizer,data_set_list):
    token_length_list = []
    for set_num in tqdm(range(len(total_data_list))):
        length = 0
        for data in total_data_list[set_num]:
            length = length + len(tokenizer(data,truncation=True,max_length=1024)["input_ids"])-2
        token_length_list.append(length)
        
    rank_legnth_dict = {}
    for num in range(len(token_length_list)):
        temp_dict = {}
        for i in range(len(token_length_list)):
            temp_dict[data_set_list[i]] = abs(token_length_list[i] - token_length_list[num])
        temp_list = []
        for i in sorted(temp_dict.items(),key=(lambda x:x[1])):
            if i[0] == data_set_list[num]:
                continue
            else:
                temp_list.append(i[0])
        rank_legnth_dict[data_set_list[num]] = temp_list
    print(rank_legnth_dict)
    save_dict(rank_legnth_dict,"rank_length")
    
def rank_cosine(total_data_list,data_set_list):
    rank_cosine_dict = {}
    count=0
    for data_list in total_data_list:
        score_dict = {}
        num=0
        for s_data_list in total_data_list:
            if data_list == s_data_list:
                num+=1
                continue
            score = 0
            for data in tqdm(data_list):
                each_split_data = set(data.split())
                for s_data in s_data_list:
                    s_each_split_data = set(s_data.split())
                    score += len(each_split_data & s_each_split_data)/(math.sqrt(len(each_split_data)*len(s_each_split_data)))
            score_dict[data_set_list[num]] = score
            num += 1
        temp_list = []
        for i in sorted(score_dict.items(),key=(lambda x:x[1]),reverse=True):
            temp_list.append(i[0])
        print(temp_list)
        rank_cosine_dict[data_set_list[count]] = temp_list
        count +=1
    save_dict(rank_cosine_dict,"rank_cosine")

def rank_precision(total_data_list,data_set_list):
    rank_rouge_dict = {}
    count=0
    for data_list in total_data_list:
        score_dict = {}
        num=0
        for s_data_list in total_data_list:
            if data_list == s_data_list:
                num+=1
                continue
            score = 0
            for data in tqdm(data_list):
                each_set_data = set()
                each_split_data = data.split()
                for idx in range(len(each_split_data)-1):
                    each_set_data.add(tuple(each_split_data[idx:idx+2]))
                for s_data in s_data_list:
                    s_each_set_data = set()
                    s_each_split_data = s_data.split()
                    for s_idx in range(len(s_each_split_data)-1):
                        s_each_set_data.add(tuple(s_each_split_data[s_idx:s_idx+2]))
                    score += (len(each_set_data & s_each_set_data)/len(s_each_set_data))
            score_dict[data_set_list[num]] = score
            num += 1
        temp_list = []
        for i in sorted(score_dict.items(),key=(lambda x:x[1]),reverse=True):
            temp_list.append(i[0])
        print(temp_list)
        rank_rouge_dict[data_set_list[count]] = temp_list
        count +=1
    save_dict(rank_rouge_dict,"rank_precision")
      
if __name__ == "__main__":
    main()