# -*- coding: utf-8 -*-
'''
Created on: 2022-04-25 15:12:29
@LastEditTime: 2022-04-25 16:00:44
@Author: fduxuan

@Desc:  训练

'''
from util import DataSet, logging
from transformers import AutoTokenizer
from tqdm import trange, tqdm


class TrainMrc:
    """ 训练MRC """
    
    def __init__(self):
        self.train_data = []
        self.dev_data = []
        self.model_id = 'bert-base-uncased'
        self.tokenizer = AutoTokenizer.from_pretrained(self.model_id)
        self.max_length = 384
        self.stride=157
    
    @staticmethod    
    def info(msg):
        logging.info(msg)
        
    def encode(self, item: dict) -> list:
        """进行encode

        Args:
            item (dict): squad_item: {id, question, answer_text, answer_start, context, is_impossible}
        return:
            list 元素为 {input_ids, attention_mask, token_type_ids, answer_text, answer(start, end)}       
        """
        data_list = []  
        tokens = self.tokenizer(
            item['context'],
            item['question'],
            max_length=self.max_length,
            return_overflowing_tokens=True,
            stride=self.stride, # 重合
            return_offsets_mapping=True,
            return_token_type_ids=True
        )
        # 先进行attention mask 手动扩充,如果长度不满max_length
        for i, offset_mapping in enumerate(tokens['offset_mapping']):
            input_ids = tokens['input_ids'][i] + [0] * (self.max_length - len(offset_mapping))
            token_type_ids = tokens['token_type_ids'][i] + [1] * (self.max_length - len(offset_mapping))
            attention_mask = [1]*len(offset_mapping) + [0] * (self.max_length - len(offset_mapping))
            answer = [0, 0]  # start, end (闭区间)
            if not item['is_impossible']:
                # 有答案
                start = item['answer_start']
                end = item['answer_start'] + len(item['answer_text'])
                res = []
                for index, pos in enumerate(offset_mapping):
                    if pos[0] < start:
                        continue
                    elif start <= pos[0] and end >= pos[1]:
                        res.append(index)
                    elif end <= pos[0]:
                        break 
                if len(res):
                    answer[0] = res[0]
                    answer[1] = res[-1]  
            data_list.append(dict(
                input_ids=input_ids,
                token_type_ids=token_type_ids,
                attention_mask=attention_mask,
                answer=answer,
                answer_text=item['answer_text']
            ))
            # 查看输出答案是否正确
            # print(item['answer_text'])
            # print(self.tokenizer.convert_ids_to_tokens(input_ids[answer[0]: answer[1]+1]))
            # print()
        return data_list
            
    def run(self):
        
        # 1. 加载数据集
        self.info('1. 初始化训练数据')
        data = DataSet().load_dataset()
        self.train_data = data['train']
        self.dev_data = data['dev']
        
        # 2. encode，切片，并且手动padding
        self.info('2. encode训练集')
        train_encode_data = []
        for item in tqdm(self.train_data[:10]):
            train_encode_data += self.encode(item)


if __name__ == "__main__":
    t = TrainMrc()
    t.run()