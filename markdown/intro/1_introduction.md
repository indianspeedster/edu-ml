::: {.cell .markdown}
# Re: Impact of Data Augmentation on Full few shot learning for intent classification. 

***
:::

::: {.cell .markdown}
## Introduction

The research paper **Impact of Data Augmentation on Full few shot learning for intent classification** centers around evaluating the effects of employing a particular data augmentation strategy on intent classification using full few-shot learning. The author focuses on three different machine learning models trained under three different scenarios. 1) Full data 2) Full few shot data (10 samples of each intent) 3) Full few shot data + Augmented data (20 samples of each intent).
Before moving ahead let's understand some of the terminologies which will be often used

:::


::: {.cell .markdown}
## Claims of the paper:
The above paper claims that "There is a minor improvement while training full few-shot learning models with data augmentation." To justify the claim the paper has this table in the result section.

<div align="center">
  <table>
    <tr>
      <th>Model</th>
      <th>Accuracy</th>
    </tr>
    <tr>
      <td>Full dataset</td>
      <td>92.75</td>
    </tr>
    <tr>
      <td>Full few-shot dataset</td>
      <td>83</td>
    </tr>
    <tr>
      <td>Full few-shot dataset + Augmented dataset</td>
      <td>84.5</td>
    </tr>
  </table>
  </div>

Further we will follow the methodology given in the paper and reproduce the results to validate the claims made in the paper.

:::


::: {.cell .markdown}
## Requirements

Before going ahead we need to install some of the libraries which we are going to use during the course of reproducing the results of this paper.

-   [transformers](https://pypi.org/project/transformers/)

-   [datasets](https://pypi.org/project/datasets/)

-   [accelerate](https://pypi.org/project/accelerate/)

-   [nltk](https://pypi.org/project/nltk/)

***
:::

::: {.cell .code}
``` python
!pip install transformers
!pip install datasets
!pip install accelerate -U
!pip install nltk
```
:::

## Next Steps
