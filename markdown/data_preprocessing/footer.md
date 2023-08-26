::: {.cell .markdown}
## Next steps

Now we are done with data preprocessing, data augmentation and tokenization. Now our data is ready to be fed to the model for training. But befor that we need to set up the training parameters and hyper parameters for our model.

When we see the paper, the paper talked about the hyperparameters such as epochs and batch size but for rest they said they used standard hyperparameters for Bert Large models. But when we researched we found that there is no specific optimizer for BERT large models, it depends upon task. so for our Classification task we were left with many choices but when it comes to picking two standard optimizer for this task we are left with two below mentioned options:

- [SGD (Stochastic Gradient Descent)](/): It is an optimization algorithm used to train machine learning models by iteratively adjusting model parameters using randomly selected mini-batches of data, with the goal of minimizing the loss function.

- [Adamw](/) : AdamW optimization is a stochastic gradient descent method that is based on adaptive estimation of first-order and second-order moments with an added method to decay weights per the techniques discussed in the paper, 'Decoupled Weight Decay Regularization' by Loshchilov, Hutter et al., 2019.


 Considering the above mentioned 2 choices we have created two notebooks, The first notebook uses AdamW and the second notebook uses SGD. You can pick your own choice and see what are the results :

-   [Notebook(Tokenization + Adamw as optimizer)](/)

-   [Notebook(Tokenization + SGD as optimizer)](/)

***
:::