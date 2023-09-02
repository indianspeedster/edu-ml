::: {.cell .markdown}
## Data Augmentation, Preprocessing and Tokenization

If you are on this notebook then you have made a choice in the previous section to go ahead with floor value while calculating n.
Let's understand this data augmentation strategy in depth with an example,
since alpha is a constant , we will fix alpha as 0.75

example 1: "what do you know about Fringe in Edinburgh next year?"
step1 : count the length of total words without stopwords. So we have l = 7
step2 : calculate n using n = alpha * l, substituting the value of n and alpha we have n = 0.75*7, n = 5.25. 
step3 : Since we choosed to go with the floor value for n, we will take n = 5.
step4 : we will pick 5 random words and then subsitute them with their synonyms.

Now, what difference would it make if we pick the ceil value instead of the floor ?

So, when we see our data, the speech text is short where the number of words lies in the range (4, 16) so one word can make a difference in the model, It is important note that ceil value or floor value while calculating "n" may create an impact in the final results.

Let's implement the augmentation strategy with ceil value of "n" while calculating n = alpha*l
:::

::: {.cell .markdown}
## Implementation
:::

::: {.cell .markdown}
### Importing relevent libraries
:::

::: {.cell .code }
``` python
import pandas as pd
import nltk
nltk.download('stopwords')
nltk.download('wordnet')
from nltk.corpus import stopwords
from nltk.corpus import wordnet
import random
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
import pickle
from transformers import AutoTokenizer
import datasets
from tqdm import tqdm
import math
seed=123
```

:::

::: {.cell .markdown}
### Data Augmentation function
:::

::: {.cell .code }
``` python
def augmentation(sentence, alpha=0.75 ):
  sentence = sentence.split(" ")
  word_index = [i for i in range(len(sentence)) if sentence[i].lower() not in stop_words]
  n = math.ceil(alpha*len(word_index))
  n_random = random.sample(word_index, n)
  for num in n_random:
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

::: {.cell .markdown}

Now, we are done with our data augmentation function, next we will work on data preprocessing, applying data augmentation strategy and  tokenizing the data which will make our data ready to be trained.

***

:::

::: {.cell .markdown}


From the last Notebook we obtained train.csv and test.csv file, these files contains raw data and the data preprocessing step will act as a bridge between raw data and the final result which we are focused to achieve. Data preprocessing can have a major impact on the final results so it's crucial to thoroughly understand what the authors did and how you could follow the same strategy.

From our understanding of the paper, The author did these three major steps:

- Encoding the labels: A machine learning model always needs a number as input instead of raw text data, so label encoding is a crucial step here.

- Sampling for full few shot Learning : The full few shot setup requires 10 samples for each label, so the author randomly took 10 samples from the dataset. it's important to make sure that you are picking unique samples. Since the author is experimenting on a 3 fold data so we will pick 30 samples and make 3 fold dataset with 10 samples each.

- Data Augmentation : The next step which author followed was implementing the data augmentation strategy and then apply the same of the previously selected 10 samples for each intent. Again the augmentation will be applied on 3 fold dataset.

- Tokenization : The last step in this notebook will be to tokenize the data, since machine learning model only understands numerical form of data, so it's necessary to make the data in the form of tokens and this process is known as tokenization.

Once we are done with the above 4 steps we will store the data and make it available for use in the next part of Reproducibility.

***
:::

::: {.cell .markdown}
### Loading the train and test data
:::

::: {.cell .code}
``` python
train_data = pd.read_csv("train.csv")
test_data = pd.read_csv("test.csv")
```
:::

::: {.cell .markdown}
### Encode the labels
:::

::: {.cell .code}
``` python
le=LabelEncoder()
train_data['intent']=le.fit_transform(train_data['intent'])
test_data['intent']=le.transform(test_data['intent'])
train_data = train_data.drop("Unnamed: 0", axis=1)
```
:::

::: {.cell .markdown}
### Split the training data to train and validation
we are again spliting the data into train and validation as when we train our model the best model while training will be selected on the basis of the validation data accuracy. ie(The model will be considered as the best model on a specific epoch when that epoch has the highest validation accuraccy.)
:::

::: {.cell .code}
``` python
df_train,val_data=train_test_split(train_data,test_size=0.10 ,random_state=seed, shuffle=True)
```
:::

::: {.cell .markdown}
**Get random 30 samples from training data**
:::

::: {.cell .code}
``` python
# Getting the unique intent
unique_labels = df_train['intent'].unique()
# Creating an empty dataframe to store all the values
sampled_df = pd.DataFrame()
#Iterating through each label and take random 30 samples from it
for label in unique_labels:
    label_df = df_train[df_train['intent'] == label]
    samples = label_df.sample(n=30, random_state=seed)
    sampled_df = sampled_df.append(samples)
sampled_df.reset_index(drop=True, inplace=True)
```
:::

::: {.cell .markdown}
**create 3 unique 10-shot dataset from previous sampled data**
:::

::: {.cell .code}
``` python
df = sampled_df
df['sampled'] = False
training_datasets = []

for i in range(3):
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
    dataset = dataset.drop("sampled",axis=1)
    dataset = dataset.reset_index(drop=True)
    training_datasets.append(dataset)


```
:::

::: {.cell .code}
``` python
train_data_full = train_data
```
:::

::: {.cell .markdown}
### Augmentating data

:::

::: {.cell .code }
``` python
#loading stop words
stop_words = set(stopwords.words('english'))
```
:::

::: {.cell .markdown}
**Apply data augmentation on each of the three training datasets**
:::

::: {.cell .code }
``` python
augmented_datasets = []
for train_data in training_datasets:
  augmented_data = train_data.copy()
  augmented_data["speech_text"] = augmented_data["speech_text"].apply(augmentation, alpha=0.6)
  augmented_data = pd.concat([train_data, augmented_data])
  augmented_datasets.append(augmented_data)
```
:::

### Tokenization

Tokenization is a fundamental process in natural language processing that plays an important role in results that any of the language model produces. All the major language models have their specific tokenizer. Since the author of the paper used Bert Large Uncased so for our reproducibility process by default we have only one choice of tokenizer. Below are some of the specific task that Bert Large tokenizer will perform:

- Text Segmentation

- Vocabulary Mapping

- Subword Tokenization

- Special Tokens 

In the next section of this notebook we will be implementing the tokenization step.

:::


::: {.cell .markdown}
**Setting up tokenizer**
:::

::: {.cell .code}
``` python
pre_trained_BERTmodel='bert-large-uncased'
BERT_tokenizer=AutoTokenizer.from_pretrained(pre_trained_BERTmodel)
```
:::

::: {.cell .markdown}
We have below mentioned 5 dataset to tokenize

- *training_datasets* : This contains a list which has 3 full few shot data of 10 samples each.

- *val_data* : This contains data in the form of a pandas dataframe which will be used as validation data while training the model.

- *test_data* : This contains data in the form of a pandas dataframe whille be used as test data for the model.

- *augmented_datasets* : This contains a list of 3 pandas datafram where each dataframe has 20 samples which contains 10 original and 10 augmented version of the original data.

- **train_data_full** : This contains a pandas data frame with entire training data.
:::


::: {.cell .markdown}
**Function to tokenize the data**
:::

::: {.cell .code}
``` python
def tokenize_data(example):
    encoded_input = BERT_tokenizer(example["speech_text"], padding="max_length", truncation=True)
    return {"input_ids": encoded_input["input_ids"], "attention_mask": encoded_input["attention_mask"], "labels": example["intent"]}
```
:::

::: {.cell .markdown}
**Tokenizing non augmented training data**
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
**Tokenizing augmented training data**
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
**Tokenizing validation data**
:::

::: {.cell .code}
``` python
val_data = datasets.Dataset.from_pandas(val_data)
val_data = val_data.map(tokenize_data)
```

:::

::: {.cell .markdown}
**Tokenizing test data**
:::

::: {.cell .code}
``` python
testdataset = datasets.Dataset.from_pandas(test_data)
test_dataset = testdataset.map(tokenize_data)
```
:::

::: {.cell .markdown}
**Tokenize full train dataset**
:::

::: {.cell .code}
``` python
train_data_full = datasets.Dataset.from_pandas(train_data_full)
train_data_full = train_data_full.map(tokenize_data)
```

:::


::: {.cell .markdown}
### Store tokenized data
:::

::: {.cell .code}
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
```
:::

::: {.cell .markdown}
### Output of this Notebook

This notebook will generate 4 files as mentioned below :

-   training_datasets.pkl

-   val_data.pkl

-   test_data.pkl

-   augmented_datasets.pkl
:::

::: {.cell .markdown}
During our data preprocessing steps we made sure the three things cleaning, transforming, and organizing data before it's fed into a model. It's important to follow the exact same preprocessing pipeline to ensure that the data is consistent and prepared in the same way as in the original study. If we have made a wrong assumption then it would lead to a different outcome and inaccurate results.

***
:::


