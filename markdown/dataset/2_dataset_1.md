::: {.cell .markdown}
## Dataset

If you are on this notebook then you have made a choice in the previous section to go ahead with **CrossValidation-Data**.
Let's understand in more depth what is inside this dataset directory.

The CrossValidation-Data directory contains these 4 folder:


- autoGeneFromRealAnno/autoGene_2018_03_22-13_01_25_169/CrossValidation

- out4ApiaiReal/Apiai_trainset_2018_03_22-13_01_25_169/CrossValidation

- out4LuisReal/Luis_trainset_2018_03_22-13_01_25_169/CrossValidation

- out4RasaReal/rasa_json_2018_03_22-13_01_25_169_80Train/CrossValidation

- out4WatsonReal/Watson_2018_03_22-13_01_25_169_trainset/CrossValidation


The first directory (autoGeneFromRealAnno/:) contains generated trainset and testset from the annotated csv file. The other four subdirectories (out4ApiaiReal, out4LuisReal, out4RasaReal and out4WatsonReal) in CrossValidation/ are the converted NLU service input data for Dialogflow, LUIS, Rasa and Watson respectively. So we will only be picking the first directory for our use case.

Inside the directory "autoGeneFromRealAnno" we have 10 fold data among which we wcan choose any of the fold to train but for now we can go ahead with 1st fold. Inside First fold we have train and test directory respectively and inside the repositories we have 64 csv files which contains intent data for each of the 64 intent.

Next we will try to collect the data and store it in a pandas dataframe and then a csv file.


***
:::

::: {.cell .markdown}
## Implementation

:::

::: {.cell .markdown}
### importing libraries
:::

::: {.cell .code}
``` python
import os
import pandas as pd
import requests
from zipfile import ZipFile
```
:::

::: {.cell .markdown}
### Clone the Github repository
:::

::: {.cell .code}
``` python
repository_url = 'https://github.com/xliuhw/NLU-Evaluation-Data/archive/refs/heads/master.zip'

response = requests.get(repository_url)
with open('repository.zip', 'wb') as file:
    file.write(response.content)

with ZipFile('repository.zip', 'r') as zip_ref:
    zip_ref.extractall('repository')
```
:::

::: {.cell .markdown}
### Arranging the relevant data

There are two subfolders inside the repository (train and test) and these folder contains many csv files with name as intent.csv where intent is the different types of intents.

We will be looping through all the csv files and then create a single file which would contain all the data.
:::

::: {.cell .code}
``` python
data = []
for folder in ["trainset", "testset/csv"]:
  csv_files = [file for file in os.listdir(f'repository/NLU-Evaluation-Data-master/CrossValidation/autoGeneFromRealAnno/autoGene_2018_03_22-13_01_25_169/CrossValidation/KFold_1/{folder}') if file.endswith('.csv')]
  merged_df = pd.DataFrame()
  for csv_file in csv_files:
      file_path = f'repository/NLU-Evaluation-Data-master/CrossValidation/autoGeneFromRealAnno/autoGene_2018_03_22-13_01_25_169/CrossValidation/KFold_1/{folder}' '/' + csv_file
      df = pd.read_csv(file_path,delimiter=";")
      merged_df = pd.concat([merged_df, df], ignore_index=True)
  data.append(merged_df)
```
:::

::: {.cell .markdown}
### Extracting the relevent columns and then saving the dataframes to a csv file
:::

::: {.cell .code}
``` python
for i, merged_df in enumerate(data):
  merged_df["merged"] = merged_df["scenario"] + "_" + merged_df["intent"]
  merged_df = merged_df[["answer_from_user", "merged"]]
  merged_df.columns = ["speech_text","intent"]
  if i == 0:
    merged_df.to_csv('train.csv')
  else:
    merged_df.to_csv('test.csv')
```
:::

::: {.cell .markdown}
The above cell will produce two csv files as output.

-   train.csv

-   test.csv

***
:::


