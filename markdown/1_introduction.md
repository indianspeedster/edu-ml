::: {.cell .markdown}
## Reproducing the paper "Full few shot learning for intent classification." {#reproducing-the-paper-full-few-shot-learning-for-intent-classification}
:::

::: {.cell .markdown}
### Few shot learning
Few-shot learning is an adaptive paradigm that empowers models to excel even in scenarios with minimal training examples. By leveraging prior knowledge from related tasks or domains, these models swiftly learn new tasks with a handful of examples. Employing transferable knowledge, meta-learning, or data augmentation, few-shot learning unlocks remarkable generalization capabilities. It proves invaluable in practical applications, ranging from medical diagnosis to natural language understanding. With the ability to extrapolate from scarce data, few-shot learning holds promise for addressing the challenges of data scarcity, enabling AI systems to swiftly adapt and thrive in various contexts, while reducing the need for massive datasets and extensive training.
:::

::: {.cell .markdown}
### Full few shot learning
Full few-shot learning enhances models' abilities with small data. Unlike standard few-shot learning that uses a few examples per task, full few-shot learning gets more examples for each task. This helps models understand tasks better and work well with limited data. It's like learning from a complete picture instead of just a small part. This approach makes models smarter across different tasks, from recognizing pictures to understanding language. By learning from a broader perspective, full few-shot learning solves the problem of having not much data and helps AI do a great job even with a small amount of information.
:::

::: {.cell .markdown}
### Intent classification
Intent classification is like teaching a computer to understand what someone wants. Imagine you're chatting with a virtual assistant, and you say, "Set an alarm for 7 AM." Intent classification helps the computer figure out that your intention is to set an alarm. It's about training the computer to recognize different purposes or intentions behind what people say. This skill is essential for making chatbots and voice assistants work better. By learning from lots of examples, the computer becomes good at identifying different intentions, making conversations with machines more helpful and natural.
:::

::: {.cell .markdown}
### BERT (Bidirectional Encoder Representation for Transformers)
BERT, short for Bidirectional Encoder Representation for Transformers, is a groundbreaking language model. Imagine a computer program that understands sentences not just word by word, but also by considering the words before and after. This two-way thinking helps BERT grasp deep language meanings and patterns. It's like teaching a computer to read between the lines. This superpower makes BERT awesome for many language tasks, from understanding texts to answering questions. It's pre-trained on tons of data, so it already knows a lot about language. This makes it versatile and powerful, transforming how computers understand and work with human language.
:::

::: {.cell .markdown}
### Requirements

Below are some of the libraries that are mandatory to install.

-   [transformers](https://pypi.org/project/transformers/)

-   [datasets](https://pypi.org/project/datasets/)

-   [accelerate](https://pypi.org/project/accelerate/)

-   [nltk](https://pypi.org/project/nltk/)
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

[HWU64](https://github.com/xliuhw/NLU-Evaluation-Data/) dataset is used by the paper we are going to reproduce and it contains natural language data for human-robot interaction in home domain which was collected and annotated for evaluating NLU Services/platforms.

The above github link contains

-   [Collected-Original-Data](https://github.com/xliuhw/NLU-Evaluation-Data/tree/master/Collected-Original-Data)

-   [CrossValidation-Data](https://github.com/xliuhw/NLU-Evaluation-Data/tree/master/CrossValidation/autoGeneFromRealAnno/autoGene_2018_03_22-13_01_25_169/CrossValidation)

In the paper it's given that the author used a given number of training and test data but you are not sure that from which of the above data they obtained that number.

Below we have 2 notebooks that have code to load data from any of all three datafolder and you can give a try to both and see through which data we are able to get equivalent results to what the author mentioned.

-   [Notebook(Collected-Original-Data)](/)

-   [Notebook(CrossValidation-Data)](/)
:::
