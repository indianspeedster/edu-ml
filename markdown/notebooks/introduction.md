::: {.cell .markdown}
## Reproducing the paper "Full few shot learning for intent classification."

:::

::: {.cell .markdown}
### Few shot learning

:::

::: {.cell .markdown}
### Full few shot learning

:::

::: {.cell .markdown}
### Intent classification

:::

::: {.cell .markdown}
### BERT (Bidirectional Encoder Representation for Transformers)


:::

::: {.cell .markdown}
### Requirements
Below are some of the libraries that are mandatory to install.

- [transformers](https://pypi.org/project/transformers/) 

- [datasets](https://pypi.org/project/datasets/)

- [accelerate](https://pypi.org/project/accelerate/)

- [nltk](https://pypi.org/project/nltk/)

:::


::: {.cell .code}
```python
!pip install transformers
!pip install datasets
!pip install accelerate -U
!pip install nltk
```
:::

::: {.cell .markdown}
### Datasets

[HWU64](https://github.com/xliuhw/NLU-Evaluation-Data/) dataset is used by the paper we are going to reproduce and it contains natural language data for human-robot interaction in home domain which was collected and annotated for evaluating NLU Services/platforms.

The above github link contains 

- [AnnotatedData](https://github.com/xliuhw/NLU-Evaluation-Data/tree/master/AnnotatedData)

- [Collected-Original-Data](https://github.com/xliuhw/NLU-Evaluation-Data/tree/master/Collected-Original-Data)

- [CrossValidation-Data](https://github.com/xliuhw/NLU-Evaluation-Data/tree/master/CrossValidation/autoGeneFromRealAnno/autoGene_2018_03_22-13_01_25_169/CrossValidation)

In the paper it's given that the author used a given number of training and test data but you are not sure that from which of the above data they obtained that number.

Below we have 3 notebooks that have code to load data from any of all three datafolder and you can give a try to all the 3 and see through which data we are able to get equivalent results to what the author mentioned.

- [Notebook(AnnotedData)](/)

- [Notebook(Collected-Original-Data)](/)

- [Notebook(CrossValidation-Data)](/)
:::