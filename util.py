# -*- coding: utf-8 -*-
'''
Created on: 2022-04-25 09:58:28
@LastEditTime: 2022-05-04 10:45:40
@Author: fduxuan

@Desc:  

'''
import json
from tqdm import tqdm
import logging
import torch
import numpy as np
import random
import os
import pymongo
mongo_client = pymongo.MongoClient('mongodb://127.0.0.1:27017')

db = mongo_client.get_database('arl')

logging.basicConfig(level=logging.INFO,
                    format='\033[1;36m%(asctime)s %(filename)s\033[0m \033[1;33m[line:%(lineno)d] \033[0m'
                           '%(levelname)s %(message)s',
                    datefmt='%Y-%m-%d %A %H:%M:%S')

class DataSet:
    
    def read(self, file: str):
        """ read file  --->
            {
                id , question, answer_text, context, answer_start, is_impossible
            }

        Args:
            file (str): 'train-v2.0.json / dev-v2.0.json'
        """
        print(f'开始读取{file}')
        squad_item = []
        count_true = 0
        count_false = 0
        data = json.load(open(file, 'r'))
        print(f"version: {data['version']}")
        for document in tqdm(data['data']):
            for paragraph in document['paragraphs']:
                    context = paragraph['context']
                    for question in paragraph['qas']:
                        if not question['is_impossible']:
                            squad_item.append(dict(
                                id=question['id'],
                                question=question['question'],
                                answer_text=question['answers'][0]['text'],
                                answer_start=question['answers'][0]['answer_start'],
                                context=context,
                                is_impossible=False,
                                answer_texts=[x['text'] for x in question['answers']],  # dev的答案不止一个
                                answer_starts=[x['answer_start'] for x in question['answers']],
                            ))
                            
                            count_true += 1
                        else:
                            squad_item.append(dict(
                                id=question['id'],
                                question=question['question'],
                                answer_text="",
                                answer_start=0,
                                context=context,
                                is_impossible=True,
                                answer_texts=[],
                                answer_starts=[]
                            ))
                            count_false += 1
        print(f'总计{count_false+count_true} | 有答案 {count_true} | 无答案 {count_false}')
        return squad_item
        
    def load_dataset(self):
        train_data = self.read('squad_v2/train-v2.0.json')
        dev_data = self.read('squad_v2/dev-v2.0.json')
        # train_data shuffer后取0.9
        # random.shuffle(train_data)
        train_data = train_data[: int(len(train_data)*0.9)]
        
        # 增强的文本测试
        # for d in dev_data:
        #     res = list(db['translation_dev'].find({"_id": d['id']}))
        #     if res:
        #         d['question'] = res[0]['translation_augment']
                
        return {
            'train': train_data,
            'dev': dev_data
        }
        
        
def f1_score(batch_data: dict, is_impossible: bool) -> float:
    """计算区间f1值
        区间答案型
            精确率(precision)是指预测的答案有多大比例的单词在标准答案中出现
            召回率(recall)是指标准答案中的单词有多大比例在预测答案中出现
            f1 = 2 * pre*recall/(pre+recall)
    Args:
        batch_data (dict): {answers[列表], start_logits, end_logits}
        with shape of (B, )
    Returns:
        float: default to 0.0
    """
    answers = batch_data['answers']
    start_logits = torch.nn.functional.softmax(batch_data['start_logits'], dim=1).cpu()
    end_logits = torch.nn.functional.softmax(batch_data['end_logits'], dim=1).cpu()
    
    f1 = 0.0
    cls_score = 1.0  # 计算无答案概率， 每一次取小的 ---> 最有可能有答案的cotext
    cls_num = 0
    possible_answer = []
    for i in range(len(answers)): 
        # 寻找所有可能的答案，并计算概率值排序
        start_logits_i = np.asarray(start_logits[i])
        end_logits_i = np.asarray(end_logits[i])

        if start_logits_i[0] + end_logits_i[0] < cls_score:
            cls_score = min(cls_score, start_logits_i[0]+end_logits_i[0])
            cls_num = i
        
        # 取5个可能的start/end
        start_indexes = np.argsort(start_logits_i)[-1 : -10 : -1].tolist()
        end_indexes = np.argsort(end_logits_i)[-1 : -10 : -1].tolist()

        for s in start_indexes:
            for e in end_indexes:
                if s > e:  # 起始比结束大
                    continue
                elif e -s > 50: # 太长 
                    continue 
                elif s == 0 or e == 0:  #不记录起始或者结束为0的
                    continue 
                else:
                    # answer， 第几个，得分
                    possible_answer.append({'answer': [s, e], 'num': i, 'score': start_logits_i[s]+end_logits_i[e]})
    possible_answer.append({'answer': [0, 0], 'num': cls_num, 'score': cls_score})
    possible_answer = sorted(possible_answer, key=lambda x: x['score'], reverse=True)

    # 小的右边界 - 大的左边界
    predict_start = possible_answer[0]['answer'][0]
    predict_end = possible_answer[0]['answer'][1]
        
    i = possible_answer[0]['num']
    # if is_impossible and possible_answer[0]['score'] == cls_score:
    #     return 1.0
    # 对每一个可能的答案分别计算f1:
    def exc(answer_start, answer_end, predict_start, predict_end):
        overlap = float(min(predict_end, answer_end) - max(predict_start, answer_start)+1)
        overlap = max(0.0, overlap * 1.0)
        if overlap == 0.0:
            return 0.0
        else:
            precision = overlap / float(answer_end - answer_start + 1)
            recall = overlap / float(predict_end-predict_start+1)
            f1 = float(2*precision*recall/(precision + recall))
            return f1
    for answer in answers[i]:
        f1 = max(f1, exc(answer[0], answer[1], predict_start, predict_end))
    em = 0.0
    if f1 == 1.0:
        em = 1.0
    return f1, em