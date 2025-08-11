import torch
from torch.utils.data import Dataset
import pandas as pd
import numpy as np
from sklearn.preprocessing import LabelEncoder

class GeneExpressionDataset(Dataset):
    def __init__(self, gene_expression_df,tissues, label_encoder=None):
        """
        参数:
        - gene_expression_df: 包含基因表达数据的,行为样本，列为基因的 DataFrame
        - transform: 可选的转换函数
        """
        self.gene_expression_data = gene_expression_df
        self.genes = self.gene_expression_data.columns  # 获取基因名称
        self.num_genes = len(self.genes)  # 总基因数
        self.indices = None
        assert gene_expression_df.shape[0] == len(tissues), "样本数必须与tissues列表长度一致"
        # 组织类型编码 (支持数值型标签)
        self.label_encoder = label_encoder if label_encoder is not None else LabelEncoder()
        self.tissues = self._encode_tissues(tissues)


    def __len__(self):
        # 返回样本的数量
        return len(self.gene_expression_data)

    def _encode_tissues(self, raw_tissues):
        """将文本标签转换为数值编码"""
        # 自动处理混合类型数据
        if isinstance(raw_tissues[0], (int, float)):
            return np.array(raw_tissues).astype(int)
        else:
            return self.label_encoder.fit_transform(raw_tissues)
    
    def __getitem__(self, idx):
        """
        返回给定索引 idx 的样本及其对应的已知混杂因素。
        """
        # 获取基因表达数据（假设每一行是一个样本，每一列是基因）
        gene_expression = self.gene_expression_data.iloc[idx].values  # 转为numpy数组
        gene_expression = torch.tensor(gene_expression, dtype=torch.float32)
        tissues = torch.LongTensor([self.tissues[idx]])
        return gene_expression, tissues