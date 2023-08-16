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
### Intent classification
Intent classification is like teaching a computer to understand what someone wants. Imagine you're chatting with a virtual assistant, and you say, "Set an alarm for 7 AM." Intent classification helps the computer figure out that your intention is to set an alarm. It's about training the computer to recognize different purposes or intentions behind what people say. This skill is essential for making chatbots and voice assistants work better. By learning from lots of examples, the computer becomes good at identifying different intentions, making conversations with machines more helpful and natural.
:::

::: {.cell .markdown}
### Full few shot learning
Full few-shot learning improves models' abilities with less data. Unlike standard few-shot learning that uses a few examples per task, full few-shot learning gets more examples for each task. This helps models understand tasks better and work well with limited data. It's like learning from a complete picture instead of just a small part. This approach makes models smarter across different tasks, from recognizing pictures to understanding language. By learning from a broader perspective, full few-shot learning solves the problem of having not much data and helps AI do a great job even with a small amount of information.
:::

::: {.cell .markdown}
## Claims of the paper:
The above paper claims that "There is a minor improvement while training full few-shot learning models with data augmentation." Further we will try to reproduce the paper and see if the claim is valid or not ?

:::


::: {.cell .markdown}
## Requirements

Below are some of the libraries that are mandatory to install.

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

::: {.cell .markdown}
### Datasets

As mentioned by the author that they used [HWU64](https://github.com/xliuhw/NLU-Evaluation-Data/) dataset and the sourcw of the data is [this](https://github.com/xliuhw/NLU-Evaluation-Data/) official Github repository and it contains natural language data for human-robot interaction in home domain which was collected and annotated for evaluating NLU Services/platforms.

The above github link contains

-   [Collected-Original-Data](https://github.com/xliuhw/NLU-Evaluation-Data/tree/master/Collected-Original-Data)

-   [CrossValidation-Data](https://github.com/xliuhw/NLU-Evaluation-Data/tree/master/CrossValidation/autoGeneFromRealAnno/autoGene_2018_03_22-13_01_25_169/CrossValidation)

In the paper it's given that the author used a given number of training and test data but you are not sure that from which of the above data they obtained that number.

Below we have 2 notebooks that have code to load data from any of all two datafolder and you can give a try to both and see through which data we are able to get equivalent results to what the author mentioned.

-   [Notebook(Collected-Original-Data)](/)

-   [Notebook(CrossValidation-Data)](/)
:::
