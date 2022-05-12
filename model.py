# -*- coding: utf-8 -*-
'''
Created on: 2022-04-25 15:54:12
@LastEditTime: 2022-05-05 10:30:35
@Author: fduxuan

@Desc:  模型文件

'''

from torch import dropout, nn, Tensor, tensor
import torch
from util import logging
from transformers import AutoModel
from transformers import WEIGHTS_NAME, CONFIG_NAME
import os

NET_CONFIG='net.pkl'

class MRC(nn.Module):
    
    def __init__(self, model_id='bert-base-uncased') -> None:
        """_summary_

        Args:
            model_id (str, optional): 模型. Defaults to 'bert-base-uncased'. // 也可以是文件夹
        """
        super().__init__()
        self.device = 'cuda'
        # self.device = 'cpu'
        
        self.model_id = model_id
        self.hidden_size = 768
        # self.hidden_size = 1024
        
        self.model = None  # plm模型
        self.dropout = nn.Dropout(0.2)
        self.linear = nn.Linear(self.hidden_size, 2)  # 隐藏层 --> start_pos &  end_pos
        self.load_model()
        self.to(self.device) # 转成cuda
        
    @staticmethod
    def info(msg):
        logging.info('\t' + msg+'...')
        
    @staticmethod
    def finish():
        logging.info('============finish==========')

    def tensor(self, data) -> Tensor:
        """ 自适应CUDA """
        return tensor(data).to(self.device)
    
    def load_model(self):
        """加载模型 || 从文件
        """
        self.info('加载plm')
        self.model = AutoModel.from_pretrained(self.model_id)
        if os.path.exists(f'{self.model_id}/{NET_CONFIG}'):
            # 存在torch模型checkpoint
            self.info(f'读取checkpoint | {self.model_id}/{NET_CONFIG}')
            state_dict = torch.load(f'{self.model_id}/{NET_CONFIG}')
            self.load_state_dict(state_dict)
            self.eval()
        self.finish()
        
    def forward(self, batch_data):
        """ 传播计算

        Args:
            batch_data (_type_): {input_ids, token_type_ids, attention_mask} (Batch, MaxLength)
        """
        input_ids = self.tensor(batch_data['input_ids'])
        token_type_ids = self.tensor(batch_data['token_type_ids'])
        attention_mask = self.tensor(batch_data['attention_mask'])
        # last_hidden_state = self.model(input_ids=input_ids, attention_mask=attention_mask,
                                        #    token_type_ids=token_type_ids)  
        # roberta没有tokentypeid
        last_hidden_state = self.model(input_ids=input_ids, attention_mask=attention_mask)
        last_hidden_state = self.dropout(last_hidden_state[0])
        logits = self.linear(last_hidden_state)  # 是个元组，取第一个才是真正的隐藏输出  # (Batch, MaxLength, HiddenSize) --> (Batch, MaxLength, 2)
        
        # logits = torch.sigmoid(logits)  # 激活
        start_logits, end_logits = logits.split(1, dim=-1)  # ((B, T, 1),(B, T, 1))
        start_logits = start_logits.squeeze(-1)  # (B, T)
        end_logits = end_logits.squeeze(-1)  # (B, T)
        return start_logits, end_logits 
    
    def save_model(self, checkpoint: str = 'checkpoint'):
        """save model

        Args:
            checkpoint (str, optional): _description_. Defaults to 'checkpoint'.
        """
        folder = os.path.exists(checkpoint)
        
        self.info(f'保存模型至{checkpoint}')
        if not folder:
            os.makedirs(checkpoint)
        self.info('\t保存PLM模型参数')
        torch.save(self.model.state_dict(), f"{checkpoint}/{WEIGHTS_NAME}") 
        self.model.config.to_json_file(f"{checkpoint}/{CONFIG_NAME}")   
        self.info('\t保存Torch网络参数')
        torch.save(self.state_dict(), f"{checkpoint}/{NET_CONFIG}")
        f"{checkpoint}/{WEIGHTS_NAME}"
    
    
if __name__ == "__main__":
    m = MRC('checkpoint')
    # m = MRC()
    # m.save_model()
    
        
        