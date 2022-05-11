import enum
import logging
import os
import sys
import errno
import time
import math
import torch
from dataclasses import dataclass, field
from itertools import chain
from typing import Optional
from tqdm import tqdm

from transformers import BertTokenizer, BertForMaskedLM, AdamW, get_linear_schedule_with_warmup

from transformers import BertTokenizer, BertForMaskedLM, AdamW
import torch
import torch.optim as optim
from torch.utils.data import DataLoader 

import argparse
from src.genie.data_module import RAMSDataModule

from src.genie.data import IEDataset, my_collate


#check for GPU
# device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# from transformers.trainer_utils import get_last_checkpoint
# from transformers.utils import check_min_version
# from transformers.utils.versions import require_version

########## Hyper Parameters ##########
output_dir = 'out'
with_tracking = False

batch_size = 64
n_epochs = 100
weight_decay = 0.0
learning_rate = 5e-5

lr_scheduler_type = "linear"
num_warmup_steps = 0.0
max_train_steps = 10_000
gradient_accumulation_steps = 1
with_tracking = False


########## Setup ##########

def createdirs(path):
    if not os.path.exists(os.path.dirname(path)):
        try:
            os.makedirs(os.path.dirname(path))
        except OSError as exc:
            if exc.errno != errno.EEXIST:
                raise

def log(str, f):
    print(str, file=sys.stderr)
    print(str, file=f)

#load data
train = IEDataset('preprocessed_rams/train_RAMS_oracle.jsonl')
test = IEDataset('preprocessed_rams/test_RAMS_oracle.jsonl')
val = IEDataset('preprocessed_rams/val_RAMS_oracle.jsonl')
        
train = DataLoader(train, 
    pin_memory=True, num_workers=2, 
    collate_fn=my_collate,
    batch_size=16, 
    shuffle=True)

test = DataLoader(test, 
    pin_memory=True, num_workers=2, 
    collate_fn=my_collate,
    batch_size=16, 
    shuffle=True)

val = DataLoader(val, 
    pin_memory=True, num_workers=2, 
    collate_fn=my_collate,
    batch_size=16, 
    shuffle=True)

device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
print(device)

logger = logging.getLogger('my_logger')
logging.basicConfig(filename='RAMS_ORACLE.log', filemode='a')

#tokenizer
tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
tokenizer.add_tokens(['<args>'])

model = BertForMaskedLM.from_pretrained('bert-base-uncased', output_hidden_states = True)
model.resize_token_embeddings(len(tokenizer))

num_train_steps = int(len(train) / batch_size * n_epochs)
optimizer = AdamW(model.parameters(), lr=lr)

scheduler = get_linear_schedule_with_warmup(
    optimizer, num_warmup_steps=0, num_training_steps=num_train_steps
)

model.to(device)
model.train()

for i_epoch in range(n_epochs):
    epoch_training_loss = 0
    epoch_testing_loss = 0
    for idx, data in tqdm(enumerate(train)):
        optimizer.zero_grad()
        input_ids = data['input_token_ids'].to(device)
        labels = data['tgt_token_ids'].to(device)
        attention_mask = data['input_attn_mask'].to(device)
        output_mlm = model(input_ids, attention_mask=attention_mask, labels=labels)
        loss = output_mlm['loss']
        loss.backward()
        optimizer.step()
        scheduler.step()

        epoch_training_loss += loss.item() * batch_size
        print(loss.item() * batch_size)

    #save model checkpoint
    logger.info(f'Epoch {i_epoch}: Train Loss = {epoch_training_loss / len(train)}')

    torch.save({
            'epoch': i_epoch,
            'model_state_dict': model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'scheduler_state_dict': scheduler.state_dict(),
            'loss': loss,
            }, f'RAMS_ORACLE_{i_epoch}')

    with torch.no_grad():
      for idx, data in tqdm(enumerate(test)):
        input_ids = data['input_token_ids'].to(device)
        labels = data['tgt_token_ids'].to(device)
        attention_mask = data['input_attn_mask'].to(device)
        output_mlm = model(input_ids, attention_mask=attention_mask, labels=labels)
        loss = output_mlm['loss']

        epoch_testing_loss += loss.item() * batch_size

    logger.info(f'Epoch {i_epoch}: Test Loss = {epoch_testing_loss / len(test)}')