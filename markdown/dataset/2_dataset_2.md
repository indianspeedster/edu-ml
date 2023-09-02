::: {.cell .markdown}
## Getting Data
If you are on this notebook then you have made a choice in the previous section to go ahead with **Collected-Original-Data**.
Your intuition might have said that since Bert large uncased is a case unsensitive model and annotating the data might not help to make a model better.

This directory contains collected original data with normalization for numbers/date etc which contain the pre-designed human-robot interaction questions and the user answers. Entire data is in CSV format and is stored 

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
import pandas as pd
from sklearn.model_selection import train_test_split
seed = 123
```
:::

::: {.cell .markdown}
### Fetching and storing the data
:::

::: {.cell .code}
``` python
url = "https://raw.githubusercontent.com/xliuhw/NLU-Evaluation-Data/master/Collected-Original-Data/paraphrases_and_intents_26k_normalised_all.csv"
dataframe = pd.read_csv(url, delimiter=";")
```
:::

::: {.cell .markdown}
### Extracting the relevent columns
:::

::: {.cell .code}
``` python
dataframe["merged"] = dataframe["scenario"] + "_" + dataframe["intent"]
new_df = dataframe[["answer", "merged"]]
new_df.columns = ["speech_text","intent"]
```
:::

::: {.cell .code}
``` python
train, test = train_test_split(new_df, test_size=0.10, random_state = seed)
```
:::

::: {.cell .markdown}
### Extracting exact number of samples as mentioned in the paper
:::

::: {.cell .code}
``` python
train = train.sample(9960, random_state=seed)
test = test.sample(1076, random_state=seed)
train = train.reset_index(drop=True)
test = test.reset_index(drop=True)
```
:::

::: {.cell .markdown}
### Exporting the data to csv
:::

::: {.cell .code}
``` python
train.to_csv("train.csv")
test.to_csv("test.csv")
```
:::

::: {.cell .markdown}
The above cell will produce two csv files as output.

-   train.csv

-   test.csv

***
:::



