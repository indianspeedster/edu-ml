# Reproducibility with Incomplete Methodology Descriptions

This work was done under [UCSC OSPO](https://ospo.ucsc.edu/) Summer of Reproducibility and is a part of the project **Reproducibility in Machine Learning Education** which aims to develop interactive educational materials about reproducibility in machine learning for use in graduate and undergraduate classes. 

*** 

> In machine learning research papers, sometimes the methodology description seems complete at first glance, but is actually missing some important details that are necessary in order to reproduce the result. This repository contains a fictitious machine learning research paper, and a sequence of accompanying Python notebooks to highlight various choices that can be made to fill in the gaps, and explore how these choices can impact the overall results of the research. 
>
> Our “research paper” is about the impact of data augmentation on few-shot learning for intent classification. We implemented a basic data augmentation strategy with synonym replacement using the HWU64 dataset and a BERT classifier, and the results suggest that synonym replacement as a data augmentation technique leads to only minor improvement in accuracy. 
> 
> In the fictitious paper, we left some of the methodology details ambiguous. When reproducing the results using the accompanying notebooks, the reader follows a “Choose Your Own Adventure” format, selecting a path through a tree, where each node represents ambiguous methodology details and branches out to different choices that are made at that instance. The leaf nodes will represent the final results, providing insights into the magnitude of the differences resulting from each node selection. Some of the choices that the reader makes are :
>
> - what subset of the source dataset to use.
> - some of the details of data pre-processing.
> - some of the details of the synonym replacement data augmentation strategy.
> - some training hyperparameters and the details of the hyperparameter search.

## Run This Experiment

- Read the paper [Impact of Data Augmentation on Full few shot learning for intent classification.](/paper.md)
- Reserve resources on [Chameleon](/notebook/) to reproduce the experiment described in the paper.
- Follow along with the notebooks provided to run the experiment



