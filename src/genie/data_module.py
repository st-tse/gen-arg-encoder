from copy import deepcopy
import os 
import json 
import re 
import random 
from collections import defaultdict 
import argparse 

import transformers 
from transformers import BartTokenizer, BertTokenizer
import torch 
from torch.utils.data import DataLoader 
import pytorch_lightning as pl 
import sys
import copy

from data import IEDataset, my_collate #temp remove .

MAX_LENGTH=512 
MAX_TGT_LENGTH=72
DOC_STRIDE=256 

class RAMSDataModule(pl.LightningDataModule):
    def __init__(self, args):
        super().__init__() 
        self.hparams = args 

        #modified here
        if args.model in ['gen','constrained-gen']:
            self.tokenizer = BartTokenizer.from_pretrained('facebook/bart-large')
            self.tokenizer.add_tokens([' <arg>',' <tgr>'])
        else:
            self.tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
            self.tokenizer.add_tokens(['<arg>','<tgr>','<pad>'])
        
    
    def get_event_type(self,ex):
        evt_type = []
        for evt in ex['evt_triggers']:
            for t in evt[2]:
                evt_type.append( t[0])
        return evt_type 

    def create_gold_gen(self, ex, ontology_dict,mark_trigger=True):
        '''assumes that each line only contains 1 event.
        Input: <s> Template with special <arg> placeholders </s> </s> Passage </s>
        Output: <s> Template with arguments and <arg> when no argument is found. 
        '''

        evt_type = self.get_event_type(ex)[0]
        context_words = [w for sent in ex['sentences'] for w in sent ]
        template = ontology_dict[evt_type.replace('n/a','unspecified')]['template']

        #change this too
        input_template = re.sub(r'<arg\d>', '<arg>', template) 


        space_tokenized_input_template = input_template.split(' ')
        tokenized_input_template = [] 
        
        for w in space_tokenized_input_template:
            tokenized_input_template.extend(self.tokenizer.tokenize(w, add_prefix_space=True))
        
        #modified here
        for triple in ex['gold_evt_links']:
            trigger_span, argument_span, arg_name = triple 
            arg_num = ontology_dict[evt_type.replace('n/a','unspecified')][arg_name]
            arg_text = ' '.join(context_words[argument_span[0]:argument_span[1]+1])

            template = re.sub('<{}>'.format(arg_num),arg_text , template)
 

        trigger = ex['evt_triggers'][0]
        if mark_trigger:
            trigger_span_start = trigger[0]
            trigger_span_end = trigger[1] +2 # one for inclusion, one for extra start marker 
            prefix = self.tokenizer.tokenize(' '.join(context_words[:trigger[0]]), add_prefix_space=True) 
            tgt = self.tokenizer.tokenize(' '.join(context_words[trigger[0]: trigger[1]+1]), add_prefix_space=True)
            
            suffix = self.tokenizer.tokenize(' '.join(context_words[trigger[1]+1:]), add_prefix_space=True)
            context = prefix + [' <tgr>', ] + tgt + [' <tgr>', ] + suffix 
        else:
            context = self.tokenizer.tokenize(' '.join(context_words), add_prefix_space=True)

        # also need to change this for pad

        #removes numbers
        output_template = re.sub(r'<arg\d>','<arg>', template ) 
        space_tokenized_template = output_template.split(' ')

        tokenized_template = [] 
        for w in space_tokenized_template:
            tokenized_template.extend(self.tokenizer.tokenize(w, add_prefix_space=True))

        return tokenized_input_template, tokenized_template, context
    
    def create_gold_oracle(self, ex, ontology_dict,mark_trigger=True):
        '''assumes that each line only contains 1 event.
        Input: <s> Template with special <arg> placeholders </s> </s> Passage </s>
        Output: <s> Template with arguments and <arg> when no argument is found. 
        '''

        evt_type = self.get_event_type(ex)[0]
        context_words = [w for sent in ex['sentences'] for w in sent ]
        template = ontology_dict[evt_type.replace('n/a','unspecified')]['template']

        input_template = re.sub(r'<arg\d>', '', template) 
        input_template = input_template.split(' ')
        input_template  = list(filter(lambda val: val != '', input_template))

        for triple in ex['gold_evt_links']:
            trigger_span, argument_span, arg_name = triple 
            #arg_len = argument_span[1] - argument_span[0] + 1
            arg_num = ontology_dict[evt_type.replace('n/a','unspecified')][arg_name]
            arg_text = ' '.join(context_words[argument_span[0]:argument_span[1]+1])
            #template_fill = arg_len * '<arg> '
            #template_fill = template_fill[:-1]

            template = re.sub('<{}>'.format(arg_num), arg_text , template)
            #input_template = re.sub('<{}>'.format(arg_num), template_fill , input_template)

        trigger = ex['evt_triggers'][0]
        if mark_trigger:
            trigger_span_start = trigger[0]
            trigger_span_end = trigger[1] +2 # one for inclusion, one for extra start marker 
            prefix = self.tokenizer.tokenize(' '.join(context_words[:trigger[0]])) 
            tgt = self.tokenizer.tokenize(' '.join(context_words[trigger[0]: trigger[1]+1]))
            
            suffix = self.tokenizer.tokenize(' '.join(context_words[trigger[1]+1:]))
            context = prefix + [' <tgr>', ] + tgt + [' <tgr>', ] + suffix 
        else:
            context = self.tokenizer.tokenize(' '.join(context_words))

        # also need to change this for pad
        #removes numbering
        output_template = re.sub(r'<arg\d>','<arg>', template ) 
        

        space_tokenized_template = output_template.split(' ')

        tokenized_template = [] 
        tokenized_input_template = []

       
        for w in space_tokenized_template:
            t = self.tokenizer.tokenize(w)
            tokenized_template.extend(t)

            try:
                match = (w == input_template[0])
            except:
                match = False

            if match:
                tokenized_input_template.extend(t)
                input_template = input_template[1:]
            else:
                for _ in range(len(t)):
                    tokenized_input_template.extend(self.tokenizer.tokenize('<arg>'))
  
        return tokenized_input_template, tokenized_template, context

    def create_gold_pad(self, ex, ontology_dict,mark_trigger=True):
        '''assumes that each line only contains 1 event.
        Input: <s> Template with special <arg> placeholders </s> </s> Passage </s>
        Output: <s> Template with arguments and <arg> when no argument is found. 
        '''

        evt_type = self.get_event_type(ex)[0]
        context_words = [w for sent in ex['sentences'] for w in sent ]
        template = ontology_dict[evt_type.replace('n/a','unspecified')]['template']
        if template[0] == '"':
            template = template[1:]

        pad = self.hparams.pad
        arg_pad = pad * '<arg> '
        arg_pad = arg_pad[:-1]
        input_template = re.sub(r'<arg\d>', arg_pad, template) 

        space_tokenized_input_template = input_template.split(' ')
        tokenized_input_template = [] 

        for w in space_tokenized_input_template:
            tokenized_input_template.extend(self.tokenizer.tokenize(w))
        
        arg_count = space_tokenized_input_template.count('<arg>') // pad

        arg_list = [[] for _ in range(arg_count)]

        for triple in ex['gold_evt_links']:
            trigger_span, argument_span, arg_name = triple 
            # print(triple)
            arg_num = ontology_dict[evt_type.replace('n/a','unspecified')][arg_name]
            arg_num = int(arg_num[3:])
    
            arg_text = context_words[argument_span[0]:argument_span[1]+1] 

            if arg_num <= arg_count:
                for w in arg_text:
                    arg_list[arg_num - 1].extend(self.tokenizer.tokenize(w))

        trigger = ex['evt_triggers'][0]
        if mark_trigger:
            trigger_span_start = trigger[0]
            trigger_span_end = trigger[1] +2 # one for inclusion, one for extra start marker 
            prefix = self.tokenizer.tokenize(' '.join(context_words[:trigger[0]])) 
            tgt = self.tokenizer.tokenize(' '.join(context_words[trigger[0]: trigger[1]+1]))
            
            suffix = self.tokenizer.tokenize(' '.join(context_words[trigger[1]+1:]))
            context = prefix + [' <tgr>', ] + tgt + [' <tgr>', ] + suffix 
        else:
            context = self.tokenizer.tokenize(' '.join(context_words))

        tokenized_template = [] 

        output_template = re.sub(r'<arg\d>', '<arg>', template) 
        space_tokenized_template = output_template.split(' ')

        arg_ind = 0
        for w in space_tokenized_template:
            if w != '<arg>':
                tokenized_template.extend(self.tokenizer.tokenize(w))
                count = 0
            else:
                arg_tokens = arg_list[arg_ind]
                for _ in range(pad):
                    if arg_tokens != []:
                        tokenized_template.append(arg_tokens.pop(0))
                    else:
                        tokenized_template.extend(self.tokenizer.tokenize('<arg>'))

                arg_ind += 1

        # print(tokenized_input_template)
        # print(tokenized_template)
        return tokenized_input_template, tokenized_template, context

    def load_ontology(self):
        # read ontology 
        ontology_dict ={} 
        with open('aida_ontology_cleaned.csv','r') as f:
            for lidx, line in enumerate(f):
                if lidx == 0:# header 
                    continue 
                fields = line.strip().split(',') 
                if len(fields) < 2:
                    break 
                evt_type = fields[0]
                args = fields[2:]
                
                ontology_dict[evt_type] = {
                        'template': fields[1]
                    }
                
                for i, arg in enumerate(args):
                    if arg !='':
                        ontology_dict[evt_type]['arg{}'.format(i+1)] = arg 
                        ontology_dict[evt_type][arg] = 'arg{}'.format(i+1)
        
        return ontology_dict 

    def prepare_data(self):
        if not os.path.exists('preprocessed_data'):
            os.makedirs('preprocessed_data')

        ontology_dict = self.load_ontology() 

        ind = 0
        
        for split,f in [('train',self.hparams.train_file), ('val',self.hparams.val_file), ('test',self.hparams.test_file)]:
            model_id = self.hparams.model
            if model_id == 'padded':
                model_id += f'_{self.hparams.pad}'
            with open(f,'r') as reader,  open('preprocessed_data/{}.jsonl'.format(split + f'_RAMS_{model_id}'), 'w') as writer:
                for lidx, line in enumerate(reader):
                    ex = json.loads(line.strip())

                    if self.hparams.model in ['gen', 'constrained-gen']:
                        input_template, output_template, context = self.create_gold_gen(ex, ontology_dict, self.hparams.mark_trigger)
                    elif self.hparams.model == 'oracle':
                        input_template, output_template, context = self.create_gold_oracle(ex, ontology_dict, self.hparams.mark_trigger)
                    else:
                        input_template, output_template, context = self.create_gold_pad(ex, ontology_dict, self.hparams.mark_trigger)

                    if self.hparams.model in ['gen', 'constrained-gen']:
                        input_tokens = self.tokenizer.encode_plus(input_template, context, 
                                add_special_tokens=True,
                                add_prefix_space=True,
                                max_length=MAX_LENGTH,
                                truncation='only_second',
                                padding='max_length')
                        tgt_tokens = self.tokenizer.encode_plus(output_template, 
                        add_special_tokens=True,
                        add_prefix_space=True, 
                        max_length=MAX_TGT_LENGTH,
                        truncation=True,
                        padding='max_length')
                    else:
                        #not sure if reversal here is required
                        input_tokens = self.tokenizer.encode_plus(context, input_template,
                                add_special_tokens=True,
                                add_prefix_space=True,
                                max_length=MAX_LENGTH,
                                truncation='only_first',
                                padding='max_length')
                        tgt_tokens = self.tokenizer.encode_plus(context, output_template,
                                add_special_tokens=True,
                                add_prefix_space=True, 
                                max_length=MAX_LENGTH,
                                truncation='only_first',
                                padding='max_length')

                    if (len(input_template) != len(output_template)) and (self.hparams.model in ['oracle', 'padded']):
                        print("Input template:", input_template)
                        print("Output template:", output_template)
                        ind += 1
                    else:
                        processed_ex = {
                            # 'idx': lidx, 
                            'doc_key': ex['doc_key'],
                            'input_token_ids':input_tokens['input_ids'],
                            'input_attn_mask': input_tokens['attention_mask'],
                            'tgt_token_ids': tgt_tokens['input_ids'],
                            'tgt_attn_mask': tgt_tokens['attention_mask'],
                        }
                        writer.write(json.dumps(processed_ex) + '\n')
                    

        print('Dropped:', ind)


    
    def train_dataloader(self):
        dataset = IEDataset('preprocessed_data/train.jsonl')
        
        dataloader = DataLoader(dataset, 
            pin_memory=True, num_workers=2, 
            collate_fn=my_collate,
            batch_size=self.hparams.train_batch_size, 
            shuffle=True)
        return dataloader 

    
    def val_dataloader(self):
        dataset = IEDataset('preprocessed_data/val.jsonl')
        
        dataloader = DataLoader(dataset, pin_memory=True, num_workers=2, 
            collate_fn=my_collate,
            batch_size=self.hparams.eval_batch_size, shuffle=False)
        return dataloader

    def test_dataloader(self):
        dataset = IEDataset('preprocessed_data/test.jsonl')
        
        dataloader = DataLoader(dataset, pin_memory=True, num_workers=2, 
            collate_fn=my_collate, 
            batch_size=self.hparams.eval_batch_size, shuffle=False)

        return dataloader


if __name__ == '__main__':
    parser = argparse.ArgumentParser() 
    parser.add_argument('--train-file',type=str,default='data/RAMS_1.0/data/train.jsonlines')
    parser.add_argument('--val-file', type=str, default='data/RAMS_1.0/data/dev.jsonlines')
    parser.add_argument('--test-file', type=str, default='data/RAMS_1.0/data/test.jsonlines')
    parser.add_argument('--train_batch_size', type=int, default=2)
    parser.add_argument('--eval_batch_size', type=int, default=4)
    parser.add_argument('--mark-trigger', action='store_true', default=True)
    parser.add_argument(
        "--model", 
        type=str, 
        required=True,
        choices=['gen','constrained-gen','padded','oracle']
    )
    parser.add_argument("--pad", type=int, default=1, help = "arg span for padded encoder version, default is 1 for single version")
    args = parser.parse_args() 

    print('ARGS PARSED')

    dm = RAMSDataModule(args=args)
    dm.prepare_data() 

    print('DONE')

    # training dataloader 
    # dataloader = dm.train_dataloader() 


    # print('DATALOADER:')
    # for idx, batch in enumerate(dataloader):
    #     print(batch)
    #     break 

    # val dataloader 