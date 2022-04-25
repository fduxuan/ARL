# -*- coding: utf-8 -*-
'''
Created on: 2022-04-25 09:58:28
@LastEditTime: 2022-04-25 21:51:02
@Author: fduxuan

@Desc:  

'''
import json
from tqdm import tqdm
import logging
import torch

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
        
        
def f1_score(batch_data: dict) -> float:
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
    start_logits = batch_data['start_logits']
    end_logits = batch_data['end_logits']
    _, predict_start = torch.max(start_logits, dim=1) 
    _, predict_end = torch.max(end_logits, dim=1)# 
    f1 = 0.0
    for i in range(len(answer_start)):
        # 无法回答的问题
        # 对于无法回答的问题，也是基于上述预测生成的start_logits与end_logits。
        # 在处理时，记录第一个位置的start_logit与end_logit的几率，
        # 即[CLS]对应的位置(在数据预处理convert_examples_to_features中，
        # 已将无法回答的问题对应的start_position与end_position都设为0)。
        # 当第一个位置的两个几率和与span中最有可能的两个几率和的差值大于null_score_diff_threshold
        # (可调节，默认为0) 
        if answer_start[i] == answer_end[i] == 0:
            if start_logits[i][predict_start[i]] + end_logits[i][predict_end[i]] < start_logits[i][0] + start_logits[i][0]:
                f1 += 1.0
            else:
                f1 += 0
        else:
            
            # 小的右边界 - 大的左边界
            overlap = float(min(predict_end[i], answer_end[i]) - max(predict_start[i], answer_start[i]))
            overlap = max(0.0, overlap * 1.0)
            if overlap == 0.0:
                f1 += 0.0
            else:
                precision = overlap / float((answer_end[i] - answer_start[i] + 1))
                recall = overlap / float((predict_end[i]-predict_start[i]))
                f1 += float(2*precision*recall/(precision + recall))
    return f1
    