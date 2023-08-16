::: {.cell .markdown}
## Tokenization

Tokenization is a fundamental process in natural language processing that plays an important role in results that any of the language model produces. All the major language models have their specific tokenizer. Since the author of the paper used Bert Large Uncased so for our reproducibility process by default we have only one choice of tokenizer. Below are some of the specific task that Bert Large tokenizer will perform:

- Text Segmentation

- Vocabulary Mapping

- Subword Tokenization

- Special Tokens 

In the next section of this notebook we will be implementing the tokenization step.

:::

::: {.cell .markdown}
### Importing the relevant libraries
:::

::: {.cell .code}
``` python
from transformers import AutoTokenizer
import datasets
import pickle
from tqdm import tqdm
```
:::

::: {.cell .markdown}
### Setting up tokenizer
:::

::: {.cell .code}
``` python
pre_trained_BERTmodel='bert-large-uncased'
BERT_tokenizer=AutoTokenizer.from_pretrained(pre_trained_BERTmodel)
```
:::

::: {.cell .markdown}
### Loading data
:::

::: {.cell .code}
``` python
with open('training_datasets.pkl', 'rb') as file:
    training_datasets = pickle.load(file)
with open('val_data.pkl', 'rb') as file:
    val_data = pickle.load(file)
with open('test_data.pkl', 'rb') as file:
    test_data = pickle.load(file)
with open('augmented_datasets.pkl', 'rb') as file:
    augmented_datasets = pickle.load(file)
with open('train_data_full.pkl', 'rb') as file:
    train_data_full = pickle.load(file)
```
:::

::: {.cell .markdown}
### Function to tokenize the data
:::

::: {.cell .code}
``` python
def tokenize_data(example):
    encoded_input = BERT_tokenizer(example["speech_text"], padding="max_length", truncation=True)
    return {"input_ids": encoded_input["input_ids"], "attention_mask": encoded_input["attention_mask"], "labels": example["intent"]}
```
:::

::: {.cell .markdown}
#### Tokenizing non augmented training data
:::

::: {.cell .code}
``` python
train_dataset=[]
for train_data_ in training_datasets:
  traindataset = datasets.Dataset.from_pandas(train_data_)
  train_dataset.append(traindataset.map(tokenize_data))
```

:::

::: {.cell .markdown}
#### Tokenizing augmented training data
:::

::: {.cell .code}
``` python
augmented_train_dataset=[]
for train_data_ in augmented_datasets:
  traindataset = datasets.Dataset.from_pandas(train_data_)
  augmented_train_dataset.append(traindataset.map(tokenize_data))
```

:::

::: {.cell .markdown}
#### Tokenizing validation data
:::

::: {.cell .code}
``` python
val_data = datasets.Dataset.from_pandas(val_data)
val_data = val_data.map(tokenize_data)
```

:::

::: {.cell .markdown}
#### Tokenizing test data
:::

::: {.cell .code}
``` python
testdataset = datasets.Dataset.from_pandas(test_data)
test_dataset = testdataset.map(tokenize_data)
```
:::

::: {.cell .markdown}
#### Tokenize full train dataset
:::

::: {.cell .code}
``` python
train_data_full = datasets.Dataset.from_pandas(train_data_full)
train_data_full = train_data_full.map(tokenize_data)
```

:::

::: {.cell .markdown}
## Setting up Training arguments

All the steps that we have been performing till now, we are making sure that it follows the exact same as mentioned by the author, The next important step in the process is to set up the model training arguments, different arguments may produce different results so it's important to make sure that the implementation is same as what author did. also in the last notebook we had to make assumption on using the specific optimizer, here we will be using that optimizer and see that how it effects the final result.

:::

::: {.cell .markdown}

### Training Arguments Explanation

1.  **`output_dir`**: Specifies the directory where model checkpoints and training logs will be saved.

2.  **`evaluation_strategy`**: Defines the strategy for evaluating the model during training. Here, it\'s set to \"epoch,\" meaning evaluation occurs after each epoch.

3.  **`save_strategy`**: Specifies when to save model checkpoints. In this case, it\'s set to \"epoch,\" indicating checkpoints are saved after each epoch.

4.  **`learning_rate`**: Determines the step size at which the optimizer adjusts model weights during training.

5.  **`per_device_train_batch_size`**: Specifies the batch size for training data per GPU, impacting memory usage and computational efficiency.

6.  **`per_device_eval_batch_size`**: Sets the batch size for evaluation data per GPU, affecting memory and computation during evaluation.

7.  **`num_train_epochs`**: Indicates the total number of training epochs, which are complete passes through the training dataset.

8.  **`warmup_ratio`**: Determines the ratio of warmup steps to the total number of training steps, helping the optimizer to smoothly adapt in the initial stages of training.

9.  **`weight_decay`**: Introduces L2 regularization to the optimizer, helping to prevent overfitting by penalizing large model weights.

10. **`load_best_model_at_end`**: Specifies whether to load the best model based on the chosen evaluation metric at the end of training.

11. **`metric_for_best_model`**: Specifies the evaluation metric used to determine the best model, which is set to \"accuracy\" in this case.

12. **`save_total_limit`**: Sets the maximum number of model checkpoints to keep, preventing excessive storage usage.

13. **`logging_dir`**: Designates the directory where training logs, such as training progress and performance metrics, will be stored.

14. **`optimizers`**: Specifies the optimizers used for model parameter updates during training, allowing for gradient-based optimization algorithms like AdamW, SGD and many more.
:::

::: {.cell .code}
``` python
def create_training_arguments_and_optimizer(lr):
    args = TrainingArguments(
        output_dir="./output",
        evaluation_strategy="epoch",
        save_strategy="epoch",
        learning_rate=lr,
        per_device_train_batch_size=8,
        per_device_eval_batch_size=8,
        num_train_epochs=3,
        warmup_ratio=0.1,
        weight_decay=0.001,
        load_best_model_at_end=True,
        metric_for_best_model="accuracy",
        save_total_limit=1,
        logging_dir="./logs",
    )

    optimizer = AdamW(args.model.parameters(), lr=args.learning_rate, weight_decay=args.weight_decay)

    return args, optimizer
```
:::

::: {.cell .markdown}
### Store tokenized data and Training Arguments
:::

::: {.cell .code id="rYLLCvlbNn0b"}
``` python
with open('train_dataset_tokenized.pkl', 'wb') as file:
    pickle.dump(train_dataset, file)
with open('val_data_tokenized.pkl', 'wb') as file:
    pickle.dump(val_data, file)
with open('test_data_tokenized.pkl', 'wb') as file:
    pickle.dump(test_dataset, file)
with open('augmented_train_dataset_tokenized.pkl', 'wb') as file:
    pickle.dump(augmented_train_dataset, file)
with open('train_dataset_full_tokenized.pkl', 'wb') as file:
    pickle.dump(list(train_data_full), file)
with open('function_train_args.pkl', 'wb') as f:
    pickle.dump(create_training_arguments_and_optimizer, f)
```
:::

::: {.cell .markdown}
### Output

This notebook will generate 6 files as mentioned below :

-   train_dataset_tokenized.pkl

-   val_data_tokenized.pkl

-   test_data_tokenized.pkl

-   augmented_train_dataset_tokenized.pkl

-   train_dataset_full_tokenized.pkl

-   function_train_args.pkl
:::

::: {.cell .markdown}
## Next Steps:

Now we have completed all the steps needed before training a model, The next step is Training the model and obtaining the final result. But wait, the author talked in the paper in Training and evaluation section that they did hyperparameter tuning but they never mentioned that they did it on all three models or just the largest model. Again why this doubt comes because if they are comparing then are they comparing it on the same ground or the models specific performance ?

So again now we are left with two choices

- Hyperparameter tuning on the largest model

- Hyperparameter tuining on all three models

It's upto you that what you are picking, below we have two notebook each with a different choice. Pick your own adventure and see how similar are your results.

-   [Notebook(Hyperparameter tuning on the largest model)](/)

-   [Notebook(Hyperparameter tuining on all three models)](/)


:::
