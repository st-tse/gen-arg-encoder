import torch
from tqdm.auto import tqdm
from transformers import AdamW
from transformers import BertTokenizer, BertForMaskedLM

tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
model = BertForMaskedLM.from_pretrained('bert-base-uncased', return_dict=True)

data_path = './content/the_fire_flower.txt'
with open(data_path, 'r', encoding='utf-8') as f:
    data = f.read().split('\n')

for sentence in data:
    if len(sentence) < 50:
        data.remove(sentence)

inputs = tokenizer(
    data,
    max_length=512,
    truncation=True,
    padding='max_length',
    return_tensors='pt'
)

inputs['labels'] = inputs['input_ids'].detach().clone()
random_tensor = torch.rand(inputs['input_ids'].shape)

masked_tensor = (random_tensor < 0.15)*(inputs['input_ids'] != 101)*(inputs['input_ids'] != 102)*(inputs['input_ids'] != 0)

nonzeros_indices = []
for i in range(len(masked_tensor)):
    nonzeros_indices.append(torch.flatten(masked_tensor[i].nonzero()).tolist())

for i in range(len(inputs['input_ids'])):
    inputs['input_ids'][i, nonzeros_indices[i]] = 103


class BookDataset(torch.utils.data.Dataset):
    def __init__(self, encodings):
        self.encodings = encodings
    
    def __len__(self):
        return len(self.encodings['input_ids'])

    def __getitem__(self, index):
        input_ids = self.encodings['input_ids'][index]
        labels = self.encodings['labels'][index]
        attention_mask = self.encodings['attention_mask'][index]
        token_type_ids = self.encodings['token_type_ids'][index]
        return {
            'input_ids': input_ids,
            'labels': labels,
            'attention_mask': attention_mask,
            'token_type_ids': token_type_ids
        }

dataset = BookDataset(inputs)

dataloader = torch.utils.data.DataLoader(
    dataset,
    batch_size=16,
    shuffle=True
)

device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')

model.to(device)


epochs = 2
optimizer = AdamW(model.parameters(), lr=1e-5)

model.train()

for epoch in range(epochs):
    loop = tqdm(dataloader)
    for batch in loop:
        optimizer.zero_grad()
        input_ids = batch['input_ids'].to(device)
        labels = batch['labels'].to(device)
        attention_mask = batch['attention_mask'].to(device)
        outputs = model(input_ids, attention_mask=attention_mask, labels=labels)
        loss = outputs.loss
        loss.backward()
        optimizer.step()

        loop.set_description("Epoch: {}".format(epoch))
        loop.set_postfix(loss=loss.item())