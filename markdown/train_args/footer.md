::: {.cell .markdown}
## Next Steps:

Now we have completed all the steps needed before training a model, The next step is Training the model and obtaining the final result. But wait, the author talked in the paper in Training and evaluation section that they did hyperparameter tuning but they never mentioned that they did it on all three models or just the largest model. Again why this doubt comes because if they are comparing then are they comparing it on the same ground or the models specific performance ?

Let's understand this in details. In our scenario, we are training three models 1) model trained on largest dataset. 2) model trained on full few shot data 3) model trained on full few shot data and it's augmented version. We are performing hyper parameter search on Learning rate as the author mentioned that they performed search on 4 standard learning rates used for training bert classifier.
Now we are confused if we need to perform search on the standard dataset and then use the same on the other two or we are performing seperately on different datasets.

So inshort we are left with two choices : 

- Hyperparameter tuning on the largest model

- Hyperparameter tuining on all three models

It's upto you that what you are picking, below we have two notebook each with a different choice. Pick your own adventure and see how similar are your results.

-   [Notebook(Hyperparameter tuning on the largest model)](/5_model_training_1.ipynb)

-   [Notebook(Hyperparameter tuining on all three models)](/5_model_training_2.ipynb)


:::