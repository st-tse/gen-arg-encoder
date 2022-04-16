from transformers import BertTokenizer, BertForMaskedLM
import torch

import argparse
from src.genie.data_module import RAMSDataModule

from torch.utils.data import DataLoader 

from src.genie.data import IEDataset, my_collate #temp remove .

tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
model = BertForMaskedLM.from_pretrained('bert-base-uncased')

dataset = IEDataset('preprocessed_rams/train_RAMS_oracle.jsonl')
        
dataloader = DataLoader(dataset, 
    pin_memory=True, num_workers=2, 
    collate_fn=my_collate,
    batch_size=16, 
    shuffle=True)

