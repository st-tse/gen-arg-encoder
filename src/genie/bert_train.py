import torch
from transformers import BertTokenizer, BertForMaskedLM

import argparse
from src.genie.data_module import RAMSDataModule

from torch.utils.data import DataLoader 

from src.genie.data import IEDataset, my_collate #temp remove .

tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
model = BertForMaskedLM.from_pretrained('bert-base-uncased')

train = IEDataset('preprocessed_rams/train_RAMS_oracle.jsonl')
test = IEDataset('preprocessed_rams/test_RAMS_oracle.jsonl')
val = IEDataset('preprocessed_rams/val_RAMS_oracle.jsonl')
        
train_dataloader = DataLoader(train, 
    pin_memory=True, num_workers=2, 
    collate_fn=my_collate,
    batch_size=16, 
    shuffle=True)
train_dataloader = DataLoader(test, 
    pin_memory=True, num_workers=2, 
    collate_fn=my_collate,
    batch_size=16, 
    shuffle=True)
val_dataloader = DataLoader(val, 
    pin_memory=True, num_workers=2, 
    collate_fn=my_collate,
    batch_size=16, 
    shuffle=True)


