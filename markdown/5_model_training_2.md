::: {.cell .markdown}
## Model training
:::

::: {.cell .code}
``` python
from transformers import TrainingArguments, Trainer
import pickle
from transformers import (
    AutoModelForSequenceClassification,
    AutoModelForSeq2SeqLM,
    AutoConfig,
    BertModel,
)
from transformers import AutoTokenizer
from sklearn.metrics import accuracy_score, precision_recall_fscore_support
import torch
import torch.nn as nn
from transformers.modeling_outputs import SequenceClassifierOutput
from transformers import AdamW
import torch.optim as optim 
```
:::

::: {.cell .markdown}
### Loading the data
:::

::: {.cell .code}
``` python
with open('train_dataset_tokenized.pkl', 'rb') as file:
    train_dataset = pickle.load(file)

with open('val_data_tokenized.pkl', 'rb') as file:
    val_dataset = pickle.load(file)

with open('test_data_tokenized.pkl', 'rb') as file:
    test_dataset = pickle.load(file)

with open('train_dataset_full_tokenized.pkl', 'rb') as file:
    train_dataset_full = [pickle.load(file)]

with open('augmented_train_dataset_tokenized.pkl', 'rb') as file:
    train_dataset_augmented = pickle.load(file)

with open('function_pickle.pkl', 'rb') as f:
    create_training_arguments_and_optimizer = pickle.load(f)
```
:::

::: {.cell .markdown}
### Setting up the training arguments
:::

::: {.cell .code}
``` python
learning_rates = [5e-5, 4e-5, 3e-5, 2e-5]

pre_trained_BERTmodel='bert-large-uncased'
BERT_tokenizer=AutoTokenizer.from_pretrained(pre_trained_BERTmodel)
```
:::

::: {.cell .markdown}
### Modifying Bert for our classification Task
:::

::: {.cell .code}
``` python
class BertModelWithCustomLossFunction(nn.Module):
    def __init__(self):
        super(BertModelWithCustomLossFunction, self).__init__()
        self.num_labels = 64
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
            loss_fct = nn.CrossEntropyLoss()
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
### Create train_model function
:::

::: {.cell .code}
``` python
def train_model(train_data, args, val_dataset, test_dataset):
    BERT_model = BertModelWithCustomLossFunction()
    optimizer = AdamW(BERT_model.parameters(), lr=args.learning_rate, weight_decay=args.weight_decay)
    trainer = Trainer(
        model=BERT_model,
        args=args,
        train_dataset=train_data,
        eval_dataset=val_dataset,
        tokenizer=BERT_tokenizer,
        compute_metrics=compute_metrics,
        optimizers=(optimizer,),
    )
    trainer.train()
    evaluation_metrics = trainer.predict(test_dataset)
    accuracy = evaluation_metrics.metrics['test_accuracy']
    torch.cuda.empty_cache()
    return accuracy
```
:::

::: {.cell .markdown}
### Setting up metrics for accuracy, precision, recall and f1
:::

::: {.cell .code}
``` python
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
``` python
import warnings
warnings.filterwarnings("ignore")
```
:::

::: {.cell .markdown}
#### Training Full dataset Model
:::

::: {.cell .code}
``` python
for train_data in train_dataset_full:
    best_accuracy = 0
    best_lr = learning_rates[0]
    for lr in learning_rates:
        args = create_training_arguments(lr)
        accuracy = train_model(train_data, args, val_dataset, test_dataset)
        if accuracy>best_accuracy:
          best_lr = lr
          best_accuracy = max(accuracy, best_accuracy)
print(f"Best Accuracy:{best_accuracy}\n Best Learning Rate: {best_lr}")
```
:::

::: {.cell .markdown}
#### Training full few shot dataset model
:::

::: {.cell .code}
``` python
best_accuracy = 0
for train_data in train_dataset:
    for lr in [best_lr]:
        args, optimizer = create_training_arguments(lr)
        accuracy = train_model(train_data, args, val_dataset, test_dataset)
        if accuracy>best_accuracy:
          best_lr = lr
          best_accuracy = max(accuracy, best_accuracy)
print(f"Best Accuracy:{best_accuracy}")
```
:::

::: {.cell .markdown}
#### Training Model on Full few shot + Augmented dataset {#training-model-on-full-few-shot--augmented-dataset}
:::

::: {.cell .code}
``` python
best_accuracy = 0
for train_data in train_dataset_augmented:
    for lr in [best_lr]::
        args = create_training_arguments(lr)
        accuracy = train_model(train_data, args, val_dataset, test_dataset)
        if accuracy>best_accuracy:
          best_lr = lr
          best_accuracy = max(accuracy, best_accuracy)
print(f"Best Accuracy:{best_accuracy}")
```
:::
