# -*- coding: utf-8 -*-
'''
Created on: 2022-04-25 15:54:12
@LastEditTime: 2022-04-25 16:10:38
@Author: fduxuan

@Desc:  模型文件

'''
from torch import nn, Tensor, tensor
import torch
from util import logging


class MRC(nn.Module):
    
    def __init__(self, model_id='bert-base-uncased') -> None:
        """_summary_

        Args:
            model_id (str, optional): 模型. Defaults to 'bert-base-uncased'.
        """
        super().__init__()
        self.device = 'cuda'
        self.model_id = model_id
    
    @staticmethod
    def info(msg):
        logging.info(msg)

    def tensor(self, data) -> Tensor:
        """ 自适应CUDA """
        return tensor(data).to(self.device)
    
    def load_model(self):
        """加载模型 || 从文件
        """
        