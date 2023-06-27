::: {.cell .markdown}
### Install relevent libraries 
:::

::: {.cell .code}
```python
!pip install transformers
!pip install datasets
!pip install accelerate -U
!pip install nltk
```
:::

::: {.cell .markdown}
### Additional step
Restart the kernel as accelerate requires restart once it's been installed 
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
import nltk
nltk.download('stopwords')
nltk.download('wordnet')
from nltk.corpus import stopwords
from nltk.corpus import wordnet
import requests
from tqdm import tqdm

from transformers import (
    AutoModelForSequenceClassification,
    AutoModelForSeq2SeqLM,
    AutoConfig,
    BertModel,
)

import torch.nn as nn
from transformers.modeling_outputs import SequenceClassifierOutput



from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
import datasets
import transformers
from transformers import BertModel, BertTokenizerFast
from transformers import AdamW, AdamWeightDecay, get_linear_schedule_with_warmup
from transformers import AutoTokenizer, TrainingArguments, Trainer
import torch
from torch import optim, nn
from torch.utils.data import DataLoader, Dataset
import torch.nn.functional as F
from statistics import median
from sklearn.metrics import accuracy_score, precision_recall_fscore_support
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
### Data Augmentation
:::

::: {.cell .code}
```python
stop_words = set(stopwords.words('english'))
```
:::

::: {.cell .markdown}
#### Function to generate random integers
:::

::: {.cell .code}
```python
import random
def generate_random_integer(n):
    numbers = list(range(n))
    random.shuffle(numbers)
    for number in numbers:
        yield number
```
:::

::: {.cell .markdown}
#### function to implement the augmentation strategy
:::

::: {.cell .code}
```python
def augmentation(sentence):
  alpha = 0.5
  sentence = sentence.split(" ")
  n = int(alpha*len(sentence))
  random_generator = generate_random_integer(len(sentence))
  random_n = []
  for _ in range(len(sentence)):
    random_number = next(random_generator)
    if sentence[random_number].lower() not in stop_words:
      random_n.append(random_number)
      if len(random_n) == n:
        break
  for num in random_n:
    word = sentence[num]
    synonyms = []
    for synset in wordnet.synsets(word):
      for synonym in synset.lemmas():
        synonyms.append(synonym.name())
    if len(synonyms)>=2:
      sentence[num] = synonyms[1]
    else:
      pass
  return " ".join(sentence)
```
:::

::: {.cell .code}
```python
augmented_data = train_data.copy()
augmented_data["intent"] = augmented_data["intent"].apply(augmentation)
train_data = pd.concat([train_data, augmented_data], axis=0)

```
:::

::: {.cell .markdown}
Plotting value counts post data augmentation

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
### Splitting the data into train and validation
:::

::: {.cell .code}
```python
df_train,df_val=train_test_split(train_data,test_size=0.10 ,random_state=seed)
```
:::

::: {.cell .markdown}
### Extracting 60 sample of each intent
:::

::: {.cell .code}
```python
unique_labels = df_train['intent'].unique()
sampled_df = pd.DataFrame()
for label in unique_labels:
    label_df = df_train[df_train['intent'] == label]
    samples = label_df.sample(n=60, random_state=seed)
    sampled_df = sampled_df.append(samples)
sampled_df.reset_index(drop=True, inplace=True)
```
:::

::: {.cell .markdown}
### Creating different dataset of 10 samples each with different data points.
:::

::: {.cell .code}
```python
df = sampled_df
df['sampled'] = False

label_counts = df['intent'].value_counts()

max_count = label_counts.max()
min_count = label_counts.min()

num_datasets = max_count // 10

training_datasets = []

for i in range(num_datasets):
    dataset = pd.DataFrame()
    for label in df['intent'].unique():
        label_df = df[(df['intent'] == label) & (df['sampled'] == False)]
        if len(label_df) >= 10:
            samples = label_df.sample(n=10)
            df.loc[samples.index, 'sampled'] = True
            dataset = pd.concat([dataset, samples])
        else:
            samples = label_df
            df.loc[samples.index, 'sampled'] = True
            dataset = pd.concat([dataset, samples])
    training_datasets.append(dataset)
val_data = df_val
```
:::

::: {.cell .code}
```python


```
:::

::: {.cell .markdown}
### Encode the labels
:::


::: {.cell .code}
```python
le=LabelEncoder()
for train_data in training_datasets:
  train_data['intent']=le.fit_transform(train_data['intent'])
val_data['intent']=le.fit_transform(val_data['intent'])
test_data['intent']=le.transform(test_data['intent'])
```
:::

::: {.cell .markdown}
### Setting up BERT Tokenizer and data loader
:::

::: {.cell .code}
```python
pre_trained_BERTmodel='bert-large-uncased'
BERT_tokenizer=AutoTokenizer.from_pretrained(pre_trained_BERTmodel)

```
:::

::: {.cell .code}
```python
def tokenize_data(example):
    encoded_input = BERT_tokenizer(example["speech_text"], padding="max_length", truncation=True)
    return {"input_ids": encoded_input["input_ids"], "attention_mask": encoded_input["attention_mask"], "labels": example["intent"]}
```
:::

::: {.cell .code}
```python
train_dataset=[]
for train_data_ in training_datasets:
  traindataset = datasets.Dataset.from_pandas(train_data_)
  train_dataset.append(traindataset.map(tokenize_data))

testdataset = datasets.Dataset.from_pandas(test_data)
test_dataset = testdataset.map(tokenize_data)

valdataset = datasets.Dataset.from_pandas(val_data)
eval_dataset = valdataset.map(tokenize_data)
```
:::

::: {.cell .code}
```python
train_dataset=[]
for train_data_ in training_datasets:
  traindataset = datasets.Dataset.from_pandas(train_data_)
  train_dataset.append(traindataset.map(tokenize_data))

testdataset = datasets.Dataset.from_pandas(test_data)
test_dataset = testdataset.map(tokenize_data)

valdataset = datasets.Dataset.from_pandas(val_data)
eval_dataset = valdataset.map(tokenize_data)
```
:::

::: {.cell .markdown}
### Setting up trainer arguments
:::

::: {.cell .code}
```python
args = TrainingArguments(
        output_dir="./output",
        evaluation_strategy="epoch",
        save_strategy="epoch",
        learning_rate=3e-5,
        per_device_train_batch_size=8 ,
        per_device_eval_batch_size=8 ,
        num_train_epochs=20,
        warmup_ratio= 0.1,
        weight_decay= 0.001,
        load_best_model_at_end=True,
        metric_for_best_model="accuracy",
        save_total_limit=1,
            )
```
:::

::: {.cell .markdown}
### Setting up Bert classifier
:::

::: {.cell .code}
```python
class BertModelWithCustomLossFunction(nn.Module):
    def __init__(self):
        super(BertModelWithCustomLossFunction, self).__init__()
        self.num_labels = len(df_train["intent"].unique())
        self.bert = BertModel.from_pretrained(
            pre_trained_BERTmodel, num_labels=self.num_labels
        )
        self.dropout = nn.Dropout(0.1)
        self.classifier = nn.Linear(1024, self.num_labels)

    def forward(self, input_ids, attention_mask, labels=None):
        outputs = self.bert(
            input_ids=input_ids,
            attention_mask=attention_mask,
        )

        output = self.dropout(outputs.pooler_output)
        logits = self.classifier(output)

        loss = None
        if labels is not None:
            # you can define any loss function here yourself
            # see https://pytorch.org/docs/stable/nn.html#loss-functions for an overview
            loss_fct = nn.CrossEntropyLoss()
            # next, compute the loss based on logits + ground-truth labels
            loss = loss_fct(logits.view(-1, self.num_labels), labels)

        return SequenceClassifierOutput(
            loss=loss,
            logits=logits,
            hidden_states=outputs.hidden_states,
            attentions=outputs.attentions,
        )
```
:::

::: {.cell .markdown}
### Setting up compute metrics
:::

::: {.cell .code}
```python
def compute_metrics(pred):
    labels = pred.label_ids
    preds = pred.predictions.argmax(-1)
    accuracy = accuracy_score(labels, preds)
    precision, recall, f1, _ = precision_recall_fscore_support(labels, preds, average='weighted')

    return {
        'accuracy': accuracy,
        'precision': precision,
        'recall': recall,
        'f1': f1
    }
```
:::

::: {.cell .markdown}
### Training the model
:::

::: {.cell .code}
```python
best_accuracy = 0
for train_dat in train_dataset:
  BERT_model = BertModelWithCustomLossFunction()
  trainer = Trainer(
        model = BERT_model,
        args = args,
        train_dataset=train_dat,
        eval_dataset=eval_dataset,
        tokenizer=BERT_tokenizer,
        compute_metrics=compute_metrics,)
  trainer.train()
  evaluation_metrics = trainer.predict(test_dataset)
  accuracy = evaluation_metrics.metrics['test_accuracy']
  best_accuracy = max(accuracy, best_accuracy)
  print(f"Best Test Accuracy for this training dataset: {accuracy}")
  torch.cuda.empty_cache()
```
:::

