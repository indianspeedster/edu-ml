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
import math
seed=123
```

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
:::

::: {.cell .code}
``` python
df_train,val_data=train_test_split(train_data,test_size=0.10 ,random_state=seed, shuffle=True)
```
:::

::: {.cell .markdown}
### Get random 30 samples from training data
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
### create 3 unique 10-shot dataset from previous sampled data
:::

::: {.cell .code}
``` python
df = sampled_df
# Create a column sample and mark it all as False and when you pick a sample mark them as True. This will make sure that you are not repeating the same sample again.
df['sampled'] = False

#creating a list to store the 10 shot dataset
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

# The output of this cell will create a list training_datasets which contains 3 10-shot dataset
```
:::

::: {.cell .markdown}
### Store data
:::

::: {.cell .code}
``` python
with open('training_datasets.pkl', 'wb') as file:
    pickle.dump(training_datasets, file)
with open('val_data.pkl', 'wb') as file:
    pickle.dump(val_data, file)
with open('test_data.pkl', 'wb') as file:
    pickle.dump(test_data, file)
with open('train_data_full.pkl', 'wb') as file:
    pickle.dump(train_data, file)
```
:::

::: {.cell .markdown}
### Data Augmentation

Upper and lower bound of n and alpha = 0.5 & 0.75
:::

::: {.cell .code }
``` python
#loading stop words
stop_words = set(stopwords.words('english'))
```
:::

::: {.cell .markdown}
#### Data Augmentation function
:::

::: {.cell .code }
``` python
def augmentation(sentence, alpha=0.5 ):
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
### Apply data augmentation on each of the three training datasets
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

::: {.cell .markdown }
### Store the augmented_data
:::

::: {.cell .code }
``` python
with open('augmented_datasets.pkl', 'wb') as file:
    pickle.dump(augmented_datasets, file)
```
:::

::: {.cell .markdown}
# Output of this Notebook

This notebook will generate 4 files as mentioned below :

-   training_datasets.pkl

-   val_data.pkl

-   test_data.pkl

-   augmented_datasets.pkl
:::
