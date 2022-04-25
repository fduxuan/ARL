# -*- coding: utf-8 -*-
'''
Created on: 2022-04-23 15:01:47
@LastEditTime: 2022-04-25 11:55:49
@Author: fduxuan

@Desc:  

'''
from transformers import AutoTokenizer
from util import DataSet

def main():
    d = DataSet()
    data = d.load_dataset()
    item = data['train'][0]
    model_id = 'bert-base-uncased'
    tokenizer = AutoTokenizer.from_pretrained(model_id, use_fast=True)
    for item in data['train']:
        if len(item['context'].split()) > 400 or item['is_impossible']:
            continue
        tokens = tokenizer(
            item['context'], item['question'], 
            return_offsets_mapping=True
        )
        # 寻找answer
        start = item['answer_start']
        end = item['answer_start'] + len(item['answer_text'])
        res = []
        for index, pos in enumerate(tokens['offset_mapping']):
            if pos[0] < start:
                continue
            elif start <= pos[0] and end >= pos[1]:
                res.append(tokens['input_ids'][index])
            elif end <= pos[0]:
                
                break
        print(item['answer_text'])
        print(tokenizer.convert_ids_to_tokens(res))
        print()
        
        
def test_strip():
    """ 测试 strip 后找到answer"""
    
    
    
if __name__ == "__main__":
    main()