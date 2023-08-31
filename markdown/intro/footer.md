::: {.cell .markdown}
### Datasets

When it comes to **reproducing** any kind of result, the role of data is incredibly crucial. If you're not using the exact same data as the original study, it becomes quite challenging to achieve precisely the same outcome. Data forms the foundation for outcomes, and even small differences in the data used can lead to variations in results. Therefore, the accuracy and similarity of data used for reproduction play a vital role in ensuring the consistency and reliability of the outcomes.

As mentioned by the author that they used [HWU64](https://github.com/xliuhw/NLU-Evaluation-Data/) dataset and the sourcw of the data is [this](https://github.com/xliuhw/NLU-Evaluation-Data/) official Github repository and it contains natural language data for human-robot interaction in home domain which was collected and annotated for evaluating NLU Services/platforms.

The above github repository contains:

**Collected-Original-Data (25K)**: collected original data with normalization for numbers/date etc which contain the pre-designed human-robot interaction questions and the user answers. They are organized in CSV format.

**AnnotatedData (25716 Lines)**: This contains annotated data for Intents and Entities, organized in csv format.
The annotated csv file has following columns: userid, answerid, scenario, intent, status, answer_annotation, notes, suggested_entities, answer_normalised, answer, question.
Most of them come from the original data collection, we keep them here for monitoring of the afterwards processing.

"answer" contains the original user answers.

"answer_normalised" were normalised from "answer".

"notes" was used for the annotators to keep a track of changes they have made.

"status" was used for annotation and post processing.

"answer_annotation" contains the annotated results which is used for generating the train/test datasets, along with "scenario", "intent" and "status".

**10-fold cross-validation** : This is 10 fold cross validation data which is formed from the AnnotatedData. This dataset was specifically designed by the authors for their specific purpose.

**Annotation Guidelines** : It contains annotation guidelines which were used to annotate the data.

So when we see the above folders, we are left with 3 different data folders, 3 because Annotation Guidelines does not cantains any data for intent classification.

When we check the content of the remaining three folders, we will see that Annotated data and 10 fold cross validation data is the same data, it's just that the 10 fold data contains 10 fold of the Annotated data. Here our intuition will say to go with one of the 10 fold data, the reason why we should choose the 10 fold data instead of the entire annotated dataset because the 1 fold data of the 10 fold data contains the same amount of data as mentioned in the paper. 

Now we are left with 2 options

1. Collected original data: 

-   [Collected-Original-Data](https://github.com/xliuhw/NLU-Evaluation-Data/tree/master/Collected-Original-Data)

-   [CrossValidation-Data](https://github.com/xliuhw/NLU-Evaluation-Data/tree/master/CrossValidation/autoGeneFromRealAnno/autoGene_2018_03_22-13_01_25_169/CrossValidation)

Now let's understand what is the difference between Annotated data and Original data through an example.

Original data: Is there an alarm for 10am?

Annotated data: "is there an alarm for ten am"

So from the above two examples we can see that in the annotated data all the text is converted to lower case. but we know that the author is using Bert Large Uncased model and The BERT "uncased" models are designed to be case-insensitive. So it's difficult to say that how much effect each of the data will bring.

From the above 2 options it's dificult to pick one choice, so we will leave it upon you to pick one and go ahead.

Below we have 2 notebooks that have code to load data from any of all two datafolder and you can give a try to both and see through which data we are able to get equivalent results to what the author mentioned.

-   [Notebook(Collected-Original-Data)](./2_dataset_2.ipynb)

-   [Notebook(CrossValidation-Data)](./2_dataset_1.ipynb)

:::
