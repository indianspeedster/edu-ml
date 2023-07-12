::: {.cell .markdown}
## Getting Data

:::

::: {.cell .markdown}
### importing libraries

:::

::: {.cell .code}
```python
import pandas as pd
from sklearn.model_selection import train_test_split

```
:::

::: {.cell .markdown}
### Fetching and storing the data

:::

::: {.cell .code}
```python
url = "https://raw.githubusercontent.com/xliuhw/NLU-Evaluation-Data/master/Collected-Original-Data/paraphrases_and_intents_26k_normalised_all.csv"
dataframe = pd.read_csv(url, delimiter=";")

```
:::

::: {.cell .markdown}
### Extracting the relevent columns

:::

::: {.cell .code}
```python
dataframe["merged"] = dataframe["scenario"] + "_" + dataframe["intent"]
new_df = dataframe[["answer", "merged"]]
new_df.columns = ["speech_text","intent"]

```
:::

::: {.cell .code}
```python
train, test = train_test_split(new_df, test_size=0.10, random_state = seed)

```
:::

::: {.cell .markdown}
### Extracting exact number of samples as mentioned in the paper

:::

::: {.cell .code}
```python
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
```python
train.to_csv("train.csv")
test.to_csv("test.csv")

```
:::