# -*- coding: utf-8 -*-
'''
Created on: 2022-04-25 15:12:29
@LastEditTime: 2022-05-07 14:15:52
@Author: fduxuan

@Desc:  训练

'''
from util import DataSet, logging, f1_score
from transformers import AutoTokenizer
from tqdm import trange, tqdm
from model import MRC
import torch
import random
import pymongo
mongo_client = pymongo.MongoClient('mongodb://127.0.0.1:27017')
db = mongo_client.get_database('arl')
class TrainMrc:
    """ 训练MRC """
    
    def __init__(self):
        self.train_data = []
        self.dev_data = []
        # self.model_id = 'bert-base-uncased'
        # self.model_id = 'albert-base-v2'
        # self.model_id = 'roberta-base'
        # self.model_id = 'google/electra-base-discriminator'
        # self.model_id = 'xlnet-base-cased'
        self.model_id = 'facebook/bart-base'
        # self.model_id = 'kssteven/ibert-roberta-base'
        
        
        # self.model_id = 'checkpoint'
        self.tokenizer = AutoTokenizer.from_pretrained(self.model_id)
        self.max_length = 386
        self.stride=128
        self.batch_size = 4
        self.epoch_num = 5
        self.learning_rate=3e-6
        self.encode_file=open('encode.txt', 'w')
        self.current_epoch = 1
                
    
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
            # item['synonym_augment'],
            max_length=self.max_length,
            return_overflowing_tokens=True,
            stride=self.stride, # 重合
            return_offsets_mapping=True,
            return_token_type_ids=True
        )
        # 先进行attention mask 手动扩充,如果长度不满max_length
        for i, offset_mapping in enumerate(tokens['offset_mapping']):
            input_ids = tokens['input_ids'][i] + [0] * (self.max_length+1 - len(offset_mapping))
            token_type_ids = tokens['token_type_ids'][i] + [1] * (self.max_length+1 - len(offset_mapping))
            attention_mask = [1]*len(offset_mapping) + [0] * (self.max_length+1 - len(offset_mapping))

            # 有答案
            answers = []
            answer_texts = [] # 多个答案
                
            answer_cls = [0, 0]
            for index, _id in enumerate(input_ids):
                if _id == self.tokenizer.cls_token_id:
                    answer_cls = [index, index]
                
            for j in range(len(item['answer_starts'])):
                answer = [0, 0]  # start, end (闭区间)
                start = item['answer_starts'][j]
                end = start + len(item['answer_texts'][j])
                res = []
                
                for index, pos in enumerate(offset_mapping):
                    if input_ids[index] == self.tokenizer.cls_token_id:
                        continue
                    elif pos[0] < start:
                        continue
                    elif start <= pos[0] and end >= pos[1]:
                        res.append(index)
                    elif end <= pos[0]:
                        break 
                if len(res):
                    answer[0] = res[0]
                    answer[1] = res[-1]  
                else:
                    answer = answer_cls
                answers.append(answer)
                answer_texts.append(item['answer_texts'][j])
            if not answers:
                answers.append(answer_cls)
                answer_texts.append("")
            data_list.append(dict(
                input_ids=input_ids,
                token_type_ids=token_type_ids,
                attention_mask=attention_mask,
                answers=answers,
                answer_texts=answer_texts,
                answer=answers[0],
                answer_text=answer_texts[0]
            ))
            # 查看输出答案是否正确
            self.encode_file.write(f"answer_text_tokens: {self.tokenizer.convert_ids_to_tokens(input_ids[answers[0][0]: answers[0][1]+1])}\n")
            self.encode_file.write(f"real_answer_text: {item['answer_text']}\n")
            self.encode_file.write('\n')
        return data_list

    def eval(self, checkpoint=None):
        """ 进行验证
        """
        if not checkpoint:
            checkpoint = f'checkpoint-{self.model_id}'
        self.info('开始验证')
        # data = DataSet().load_dataset()
        # self.train_data = data['train']
        # self.dev_data = data['dev']
        self.dev_data = list(db['augment_dev'].find())
        bar = tqdm(self.dev_data)
        mrc = MRC(checkpoint)
        mrc.eval()
        f1 = 0.0
        f1_no_answer = 0.0
        f1_has_answer = 0.0
        em = 0.0
        em_no_answer = 0.0
        em_has_answer = 0.0
        count = 0
        count_has_answer = 0
        count_no_answer = 0
        
        with torch.no_grad():
            result = {}
            for item in bar:
                # 每次只一个
                dev_encode_data = self.encode(item)
                batch_data = dict(
                    input_ids=[x['input_ids'] for x in dev_encode_data],
                    token_type_ids=[x['token_type_ids'] for x in dev_encode_data],
                    attention_mask=[x['attention_mask'] for x in dev_encode_data],
                    answers=[x['answers'] for x in dev_encode_data],
                    answer_texts=[x['answer_texts'] for x in dev_encode_data],
                )
                batch_data['start_logits'], batch_data['end_logits'] = mrc(batch_data)
                _f1, _em = f1_score(batch_data, item['is_impossible'])
                if item['is_impossible']:
                    f1_no_answer += _f1
                    em_no_answer += _em
                    count_no_answer += 1
                else:
                    f1_has_answer += _f1
                    em_has_answer += _em
                    count_has_answer += 1
                f1 += _f1
                em += _em
                count += 1
                
                if count and count_has_answer and count_no_answer:
                    result = {'f1_total': f1/count, 'f1_has_answer': f1_has_answer/count_has_answer, 'f1_no_answer': f1_no_answer/count_no_answer, 'em_total': em/count, 'em_has_answer': em_has_answer/count_has_answer, 'em_no_answer': em_no_answer/count_no_answer}
                    bar.set_postfix(result)
            logging.info(f"{result}")
    
    def run(self):
        
        # 1. 加载数据集
        self.info('1. 初始化训练数据')
        data = DataSet().load_dataset()
        self.train_data = data['train']
        self.dev_data = data['dev']
        
        # 2. encode，切片，并且手动padding
        self.info('2. encode训练集')
        train_encode_data = []
        for item in tqdm(self.train_data):
            train_encode_data += self.encode(item)
        
        # 3. 加载模型
        self.info('3. 加载模型和优化器')
        # mrc = MRC(self.model_id)
        # mrc = MRC('checkpoint')
        mrc = MRC(self.model_id)
        optimizer = torch.optim.Adam(mrc.parameters(), lr=self.learning_rate)
        loss_func = torch.nn.CrossEntropyLoss()
        self.info(f'\t optimizer = Adam || loss_func = CrossEntropyLoss || learning_rate={self.learning_rate}')
        
        # 4. 转为batch格式训练
        self.info(f'4. 开始训练 || batch_size = {self.batch_size}')
        f = open('res.txt', 'w')
        with torch.enable_grad():
            mrc.train()
            for epoch in range(0, self.epoch_num):
                self.current_epoch = epoch+1
                logging.info(f'当前第{self.current_epoch}轮')
                loss_value = 0.0
                random.shuffle(train_encode_data)
                bar = trange(0, len(train_encode_data), self.batch_size)
                for i in bar:
                    end = min(i + self.batch_size, len(train_encode_data))
                    batch_data = dict(
                        input_ids=[x['input_ids'] for x in train_encode_data[i: end]],
                        token_type_ids=[x['token_type_ids'] for x in train_encode_data[i: end]],
                        attention_mask=[x['attention_mask'] for x in train_encode_data[i: end]],
                        answer_start=mrc.tensor([x['answer'][0] for x in train_encode_data[i: end]]),
                        answer_end=mrc.tensor([x['answer'][1] for x in train_encode_data[i: end]]),
                        answer_text=[x['answer_text'] for x in train_encode_data[i: end]],
                    )
                    optimizer.zero_grad()
                    start_logits, end_logits = mrc(batch_data)
                    
                    loss = loss_func(start_logits, batch_data['answer_start']) + loss_func(end_logits, batch_data['answer_end'])
                    loss.backward()
                    optimizer.step()
                    loss_value = loss.item()
                    bar.set_postfix({'epoch': epoch+1, 'loss': loss_value})
                    
                    # 查看最大值
                    _, predict_starts = torch.max(start_logits, dim=1) 
                    _, predict_ends = torch.max(end_logits, dim=1)# 
                    for j in range(0, len(predict_starts)):
                        predict_answer = []
                        input_ids = batch_data['input_ids'][j]
                        answer_start = batch_data['answer_start']
                        answer_end = batch_data['answer_end']
                        
                        if predict_ends[j] >= predict_starts[j]:
                            predict_answer = self.tokenizer.convert_ids_to_tokens(input_ids[predict_starts[j]: predict_ends[j]+1])
                        f.write(f"answer_text': {self.tokenizer.convert_ids_to_tokens(input_ids[answer_start[j]: answer_end[j]+1])}\n")
                        f.write(f"real_answer_text': {batch_data['answer_text'][j]}\n")
                        
                        f.write(f"predict_answer_text': {predict_answer}\n")
                        f.write('\n')
                mrc.save_model(f'checkpoint-{self.model_id}/{self.current_epoch}')  
                self.eval(f'checkpoint-{self.model_id}/{self.current_epoch}')       
                                    
if __name__ == "__main__":
    t = TrainMrc()
    t.run()
    # t.eval(f'checkpoint-{t.model_id}/2')
