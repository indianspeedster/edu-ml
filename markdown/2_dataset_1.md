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

There are two subfolders inside the repository (train and test) and these folder contains many csv files with name as inten.csv where intent is the different types of intents.

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
:::
