::: {.cell .markdown}
## Getting Data
In the world of machine learning, the data you use is like the building blocks of your model. Good data can help your model perform really well and give useful results, while not-so-good data can lead to results that don't really make sense or aren't very accurate. So, having good, reliable data is super important because it's what your model learns from and it can make a big difference in how well your model works and the results it gives you.

Likewise, when it comes to **reproducing** any kind of result, the role of data is incredibly crucial. If you're not using the exact same data as the original study, it becomes quite challenging to achieve precisely the same outcome. Data forms the foundation for outcomes, and even small differences in the data used can lead to variations in results. Therefore, the accuracy and similarity of data used for reproduction play a vital role in ensuring the consistency and reliability of the outcomes.

***
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

::: {.cell .markdown}
Once we have the data ready, then we can now focus on Augmenting the data.

As discussed in the paper, the Authors are following synonym replacement strategy with a special formulae n = α * l where n is the number of words that is going to be replaced, α is a constant whose value lies between 0 and 1 and l is the length of the sentence.

Now when calculating n, there is high probability that the value will be a decimal value and since n can be only an integer, the author never specified that which value of n we are supposed to pick. ie (ceil or floor). Intent classification task has less number of words as input and even if there is difference of one word in the the augmented text due to this ceil, floor confusion, then it may lead to different results.

For the data preprocessing we have two notebooks which will focus on both the scenarios taking a ceil value for n and taking a floor value for n.

-   [Notebook(DataPreProcess_floor(n))](/)

-   [Notebook(DataPreProcess_ceil(n))](/)

:::

