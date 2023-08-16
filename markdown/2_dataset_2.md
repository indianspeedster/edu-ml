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
Once we have the data ready, then we can now focus on Augmenting the data.

As discussed in the paper, the Authors are following synonym replacement strategy with a special formulae n = α * l where n is the number of words that is going to be replaced, α is a constant whose value lies between 0 and 1 and l is the length of the sentence.

Now when calculating n, there is high probability that the value will be a decimal value and since n can be only an integer, the author never specified that which value of n we are supposed to pick. ie (ceil or floor). Intent classification task has less number of words as input and even if there is difference of one word in the the augmented text due to this ceil, floor confusion, then it may lead to different results.

For the data preprocessing we have two notebooks which will focus on both the scenarios taking a ceil value for n and taking a floor value for n.

-   [Notebook(DataPreProcess_floor(n))](/)

-   [Notebook(DataPreProcess_ceil(n))](/)

:::
