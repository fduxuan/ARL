# -*- coding: utf-8 -*-
'''
Created on: 2022-04-25 09:58:28
@LastEditTime: 2022-04-25 11:23:47
@Author: fduxuan

@Desc:  

'''
import json
from tqdm import tqdm

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