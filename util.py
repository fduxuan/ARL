# -*- coding: utf-8 -*-
'''
Created on: 2022-04-25 09:58:28
@LastEditTime: 2022-04-26 15:47:16
@Author: fduxuan

@Desc:  

'''
import json
from tqdm import tqdm
import logging
import torch
import numpy as np

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
                                is_impossible=False
                            ))
                            count_true += 1
                        else:
                            squad_item.append(dict(
                                id=question['id'],
                                question=question['question'],
                                answer_text="",
                                answer_start=0,
                                context=context,
                                is_impossible=True
                            ))
                            count_false += 1
        print(f'总计{count_false+count_true} | 有答案 {count_true} | 无答案 {count_false}')
        return squad_item
        
    def load_dataset(self):
        train_data = self.read('squad_v2/train-v2.0.json')
        dev_data = self.read('squad_v2/dev-v2.0.json')
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
        batch_data (dict): {answer_start, answer_end, start_logits, end_logits}
        with shape of (B, )
    Returns:
        float: default to 0.0
    """
    answer_start = batch_data['answer_start']
    answer_end = batch_data['answer_end']
    start_logits = torch.nn.functional.softmax(batch_data['start_logits'], dim=1)
    end_logits = torch.nn.functional.softmax(batch_data['end_logits'], dim=1)
    
    f1 = 0.0
    cls_score = 1.0  # 计算无答案概率， 每一次取小的 ---> 最有可能有答案的cotext
    possible_answer = []
    for i in range(len(answer_start)): 
        # 寻找所有可能的答案，并计算概率值排序
        start_logits_i = np.asarray( start_logits[i])
        end_logits_i = np.asarray( end_logits[i])

        cls_score = min(cls_score, start_logits_i[0]+end_logits_i[0])

        # 取5个可能的start/end
        start_indexes = np.argsort(start_logits_i)[-1 : -5 : -1].tolist()
        end_indexes = np.argsort(end_logits_i)[-1 : -5 : -1].tolist()

        for s in start_indexes:
            for e in end_indexes:
                if s > e:  # 起始比结束大
                    continue
                elif e -s > 10: # 太长 
                    continue 
                elif s == 0 or e == 0:  #不记录起始或者结束为0的
                    continue 
                else:
                    # answer， 第几个，得分
                    possible_answer.append({'answer': [s, e], 'num': i, 'score': start_logits_i[s]+end_logits_i[e]})
    possible_answer = sorted(possible_answer, key=lambda x: x['score'], reverse=True)
    if not len(possible_answer):
        if is_impossible:
            return 1.0
        else:
            return 0.0
    if cls_score >= possible_answer[0]['score']:
        # 无答案
        if is_impossible:
            return 1.0
        else: 
            return 0.0
    else:
        # 小的右边界 - 大的左边界
        predict_start = possible_answer[0]['answer'][0]
        predict_end = possible_answer[0]['answer'][1]
        
        i = possible_answer[0]['num']
        overlap = float(min(predict_end, answer_end[i]) - max(predict_start, answer_start[i])+1)
        overlap = max(0.0, overlap * 1.0)
        if overlap == 0.0:
                return 0.0
        else:
            precision = overlap / float(answer_end[i] - answer_start[i] + 1)
            recall = overlap / float(predict_end-predict_start+1)
            f1 = float(2*precision*recall/(precision + recall))
            return f1