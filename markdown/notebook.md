::: {.cell .markdown}
### Install transformers 
:::

::: {.cell .code}
```python
!pip install transformers
```
:::

::: {.cell .markdown}
### Import relevent libraries
:::

::: {.cell .code}
```python
import pandas as pd
import numpy as np
import seaborn as sns
import re
import matplotlib.pyplot as plt
from collections import defaultdict
import csv
import io
import json
import os

import requests
from tqdm import tqdm

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder

import transformers
from transformers import BertModel, BertTokenizerFast
from transformers import AdamW, AdamWeightDecay, get_linear_schedule_with_warmup

import torch
from torch import optim, nn
from torch.utils.data import DataLoader, Dataset
import torch.nn.functional as F 
seed = 1331
```
:::

::: {.cell .code}
```python
device=torch.device('cuda:0')
torch.cuda.get_device_name(0)
```
:::

::: {.cell .markdown}
### Fetch and store the data
:::

::: {.cell .code}
```python
_HEADER = ["text", "category"]
PATTERNS = {
    "train": "https://raw.githubusercontent.com/xliuhw/NLU-Evaluation-Data"
             "/master/CrossValidation/autoGeneFromRealAnno/autoGene_2018_03_"
             "22-13_01_25_169/CrossValidation/KFold_1/trainset/{f}",
    "test": "https://raw.githubusercontent.com/xliuhw/NLU-Evaluation-Data/"
            "master/CrossValidation/autoGeneFromRealAnno/autoGene_2018_03_"
            "22-13_01_25_169/CrossValidation/KFold_1/testset/csv/{f}"
}

LIST_OF_FILES = (
    'alarm_query.csv\nalarm_remove.csv\nalarm_set.csv\naudio_volum'
    'e_down.csv\naudio_volume_mute.csv\naudio_volume_up.csv\ncalend'
    'ar_query.csv\t\ncalendar_remove.csv\t\ncalendar_set.csv\t\ncoo'
    'king_recipe.csv\t\ndatetime_convert.csv\t\ndatetime_query.csv'
    '\t\nemail_addcontact.csv\t\nemail_query.csv\t\nemail_querycon'
    'tact.csv\t\nemail_sendemail.csv\t\ngeneral_affirm.csv\t\ngener'
    'al_commandstop.csv\t\ngeneral_confirm.csv\t\ngeneral_dontcare.'
    'csv\t\ngeneral_explain.csv\t\ngeneral_joke.csv\t\ngeneral_neg'
    'ate.csv\t\ngeneral_praise.csv\t\ngeneral_quirky.csv\t\ngenera'
    'l_repeat.csv\t\niot_cleaning.csv\t\niot_coffee.csv\t\niot_hue'
    '_lightchange.csv\t\niot_hue_lightdim.csv\t\niot_hue_lightoff.'
    'csv\t\niot_hue_lighton.csv\t\niot_hue_lightup.csv\t\niot_wemo_'
    'off.csv\t\niot_wemo_on.csv\t\nlists_createoradd.csv\t\nlists_'
    'query.csv\t\nlists_remove.csv\t\nmusic_likeness.csv\t\nmusic_q'
    'uery.csv\t\nmusic_settings.csv\t\nnews_query.csv\t\nplay_audio'
    'book.csv\t\nplay_game.csv\t\nplay_music.csv\t\nplay_podcasts.'
    'csv\t\nplay_radio.csv\t\nqa_currency.csv\t\nqa_definition.csv'
    '\t\nqa_factoid.csv\t\nqa_maths.csv\t\nqa_stock.csv\t\nrecomme'
    'ndation_events.csv\t\nrecommendation_locations.csv\t\nrecomme'
    'ndation_movies.csv\t\nsocial_post.csv\t\nsocial_query.csv\t\n'
    'takeaway_order.csv\t\ntakeaway_query.csv\t\ntransport_query.c'
    'sv\t\ntransport_taxi.csv\t\ntransport_ticket.csv\t\ntransport'
    '_traffic.csv\t\nweather_query.csv\t'.split())




def _get_category_rows(fname: str, set_name: str):
    pattern = PATTERNS[set_name]
    url = pattern.format(f=fname)
    request = requests.get(url)

    reader = csv.reader(
        io.StringIO(request.content.decode("utf-8")), delimiter=";"
    )
    first_row = next(reader)
    scenario_i, intent_i = first_row.index("scenario"), first_row.index(
        "intent")
    answer_i = first_row.index("answer_from_anno")

    rows = []
    for row in reader:
        text = row[answer_i]
        category = f"{row[scenario_i]}_{row[intent_i]}"
        rows.append([text, category])
    return rows


def _get_final_rows(set_name: str):
    final_rows = [_HEADER]
    for f in tqdm(LIST_OF_FILES):
        final_rows += _get_category_rows(f, set_name)
    return final_rows


def _write_data_into_file(path, rows):
    with open(path, "w") as data_file:
        writer = csv.writer(data_file, quoting=csv.QUOTE_ALL)
        writer.writerows(rows)


def _main():
    data_dir = os.getcwd()

    if not os.path.exists(data_dir):
        os.mkdir(data_dir)

    print("Getting train data")
    train_rows = _get_final_rows(set_name="train")
    _write_data_into_file(
        path=os.path.join(data_dir, "train.csv"),
        rows=train_rows
    )

    print("Getting test data")
    test_rows = _get_final_rows(set_name="test")
    _write_data_into_file(
        path=os.path.join(data_dir, "test.csv"),
        rows=test_rows
    )

    print("Creating categories.json file")
    _, train_cats = zip(*train_rows[1:])
    _, test_cats = zip(*test_rows[1:])
    categories = sorted(list(
        set(train_cats) | set(test_cats)
    ))
    with open(os.path.join(data_dir, "categories.json"), "w") as f:
        json.dump(categories, f)


if __name__ == "__main__":
    _main()

```
:::

::: {.cell .markdown}
### Load and visualize data
:::

::: {.cell .code}
```python
train_data = pd.read_csv("train.csv")
test_data = pd.read_csv("test.csv")
train_data.columns = ["speech_text","intent"]
test_data.columns = ["speech_text","intent"]
```
:::

::: {.cell .code}
```python
train_data.describe()
```
:::

::: {.cell .code}
```python
intent_counts = train_data['intent'].value_counts()
plt.figure(figsize=(10, 6))
intent_counts.plot(kind='bar')
plt.title('Intent Counts')
plt.xlabel('Intent')
plt.ylabel('Count')
plt.show()
```
:::

::: {.cell .markdown}
### Encode the labels
:::


::: {.cell .code}
```python
le=LabelEncoder()
train_data['intent']=le.fit_transform(train_data['intent'])
test_data['intent']=le.transform(test_data['intent'])
print(le.classes_)
```
:::

::: {.cell .markdown}
### Setting up BERT Tokenizer
:::

::: {.cell .code}
```python
pre_trained_BERTmodel='bert-large-uncased'
BERT_tokenizer=BertTokenizerFast.from_pretrained(pre_trained_BERTmodel)
```
:::

::: {.cell .markdown}
### Create torch dataset class
This dataset class will be used to serve data to Dataloader
:::

::: {.cell .code}
```python
class Speech_text_Dataset(Dataset):

  def __init__(self,text,intent,tokenizer,max_length):
    self.text=text
    self.intent=intent
    self.tokenizer=tokenizer
    self.max_length=max_length

  def __len__(self):
    return len(self.text)

  def __getitem__(self,item):
    text = str(self.text[item])
    intent = self.intent[item]

    encoding = self.tokenizer.encode_plus(
        text,
        max_length=Max_length,
        add_special_tokens=True,
        padding="max_length",
        truncation=True,
        return_attention_mask=True,
        return_token_type_ids=False,
        return_tensors='pt'
       )
           
    return {
        'speech_text':text,
        'input_ids':encoding['input_ids'].flatten(),
        'attention_mask':encoding['attention_mask'].flatten(),
        'intent' : torch.tensor(intent,dtype=torch.long)
    }
```
:::

::: {.cell .markdown}
### Split the training data into training and validation

:::

::: {.cell .code}
```python
df_train,df_val=train_test_split(train_data,test_size=0.15 ,random_state=seed)

print('Print the shape of datasets...')
print(f'Training dataset : {df_train.shape} ')
print(f'Testing dataset : {test_data.shape}') 
print(f'Validation dataset : {df_val.shape}')
```
:::

::: {.cell .markdown}
### Create the Dataloader
This will be used to train model and make sure that the data is being fed in batches
:::

::: {.cell .code}
```python
batch_size=32
Max_length= 50
def data_loader(df,tokenizer, max_length, batch):
  ds=Speech_text_Dataset(
      text=df.speech_text.to_numpy(),
      intent=df.intent.to_numpy(),
      tokenizer=tokenizer,
      max_length=Max_length
  )

  return DataLoader(
      ds,
      batch_size=batch_size,
      num_workers=4
  )

train_DataLoader=data_loader(df_train, BERT_tokenizer,Max_length,batch_size)
test_DataLoader=data_loader(test_data, BERT_tokenizer,Max_length,batch_size)
valid_DataLoader=data_loader(df_val, BERT_tokenizer,Max_length,batch_size)
```
:::

::: {.cell .code}
```python
BERT_data=next(iter(train_DataLoader))
input_ids = BERT_data['input_ids'].to(device)
attention_mask = BERT_data['attention_mask'].to(device)
targets=BERT_data['intent'].to(device)
print('Shape of the BERT_data keys...')
print(f"Input_ids : {BERT_data['input_ids'].shape}")
print(f"Attention_mask : {BERT_data['attention_mask'].shape}")
print(f"targets : {BERT_data['intent'].shape}")
```
:::

::: {.cell .markdown}
### Load Pre-trained BERT Model 
This Bert model will be used for training with slight modification ie, adding an extra layer at the end
:::

::: {.cell .code}
```python
BERT_model = BertModel.from_pretrained(pre_trained_BERTmodel)
BERT_model=BERT_model.to(device)
```
:::

::: {.cell .code}
```python
n_classes= len(train_data["intent"].unique())
```
:::

::: {.cell .code}
```python
class BERT_IntentClassifier(nn.Module):
   def __init__(self, n_classes):
     super(BERT_IntentClassifier, self).__init__()
     self.bert = BertModel.from_pretrained(pre_trained_BERTmodel)
     self.drop = nn.Dropout(p=0.1)
     self.out = nn.Linear(self.bert.config.hidden_size, n_classes)
   def forward(self, input_ids, attention_mask):
     _, pooled_output = self.bert(
         input_ids=input_ids,
         attention_mask=attention_mask,return_dict=False
    )
     output = self.drop(pooled_output)
     output=self.out(output)
     return output
```
:::

::: {.cell .code}
```python
BERT_model = BERT_IntentClassifier(n_classes)
BERT_model=BERT_model.to(device)
```
:::

::: {.cell .code}
```python
F.softmax(BERT_model(input_ids,attention_mask), dim=1).to(device)
```
:::

::: {.cell .markdown}
### Setting the Training hyperparameters
:::

::: {.cell .code}
```python
epochs=6
optimizer=AdamW(BERT_model.parameters(),lr=4e-5)
total_steps=len(train_DataLoader)*epochs

scheduler=get_linear_schedule_with_warmup(
    optimizer,
    num_warmup_steps=0,
    num_training_steps=total_steps
)

loss_fn=nn.CrossEntropyLoss().to(device)
```
:::

::: {.cell .markdown}
### Train and eval Function
:::

::: {.cell .code}
```python
 train(
  model,
  data_loader,
  loss_fn,
  optimizer,
  device,
  scheduler,
  n_observations
):
  model = model.train()
  losses = []
  correct_predictions = 0
  for d in data_loader:
    input_ids = d["input_ids"].to(device)
    attention_mask = d["attention_mask"].to(device)
    targets = d["intent"].to(device)
    outputs = model(
      input_ids=input_ids,
      attention_mask=attention_mask
      )
    _, preds = torch.max(outputs, dim=1)
    loss = loss_fn(outputs, targets)
    correct_predictions += torch.sum(preds == targets)
    losses.append(loss.item())
    loss.backward()   
    optimizer.step()
    scheduler.step()
    optimizer.zero_grad()
  return correct_predictions.double() / n_observations, np.mean(losses)
```
:::

::: {.cell .code}
```python
def eval_model(model, data_loader,device,loss_fn, n_observations):
  model = model.eval()
  losses = []
  correct_predictions = 0
  with torch.no_grad():
    for d in data_loader:
      input_ids = d["input_ids"].to(device)
      attention_mask = d["attention_mask"].to(device)
      targets = d["intent"].to(device)
      outputs = model(
        input_ids=input_ids,
        attention_mask=attention_mask,
      )
      _, preds = torch.max(outputs, dim=1)
      loss = loss_fn(outputs, targets)
      correct_predictions += torch.sum(preds == targets)
      losses.append(loss.item())
  return correct_predictions.double() / n_observations, np.mean(losses)
```
:::

::: {.cell .markdown}
### Finetune the model
:::

::: {.cell .code}
```python
%%time
history = defaultdict(list)
best_accuracy = 0
for epoch in range(epochs):
  print(f'Epoch {epoch + 1}/{epochs}')
  print('-' * 10)
  train_acc, train_loss = train(
    BERT_model,
    train_DataLoader,
    loss_fn,
    optimizer,
    device,
    scheduler,
    len(df_train)
  )
  print(f'Train loss {train_loss} accuracy {train_acc}')
  val_acc, val_loss = eval_model(
    BERT_model,
    valid_DataLoader,
    device,
    loss_fn,
    len(df_val)
  )
  print(f'Validation  loss {val_loss} accuracy {val_acc}')
  print()
  history['train_acc'].append(train_acc)
  history['train_loss'].append(train_loss)
  history['val_acc'].append(val_acc)
  history['val_loss'].append(val_loss)
  if val_acc > best_accuracy:
    torch.save(BERT_model.state_dict(), 'best_model_state.bin')
    best_accuracy = val_acc
```
:::

::: {.cell .markdown}

:::

::: {.cell .code}
```python

```
:::