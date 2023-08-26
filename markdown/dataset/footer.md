::: {.cell .markdown}
Once we have the data ready, then we can now focus on Augmenting the data.

As discussed in the paper, the Authors are following synonym replacement strategy with a special formulae n = α * l where n is the number of words that is going to be replaced, α is a constant whose value lies between 0 and 1 and l is the length of the sentence.
:::

::: {.cell .markdown}
Now let's understand how this synonym replacement works.

Suppose we have a sentence "is there an alarm for ten am"
Step 1 is to remove the stopwords from the sentence, so now the sentence would be "there alarm ten am".

The length of the sentence now is 4 which is l.

Let's take alpha as 0.6, So, now when we perform calculation we get n = 0.75*4, which is equal to 3, So now we will pick three random words and replace then with their synonyms. 

:::

:::{.cell .markdown}

Now when calculating n, there is high probability that the value will be a decimal value and since n can be only an integer, the author never specified that which value of n we are supposed to pick. ie (ceil or floor). Intent classification task has less number of words as input and even if there is difference of one word in the the augmented text due to this ceil, floor confusion, then it may lead to different results.

For the data preprocessing we have two notebooks which will focus on both the scenarios taking a ceil value for n and taking a floor value for n.

-   [Notebook(DataPreProcess_floor(n))](/)

-   [Notebook(DataPreProcess_ceil(n))](/)

:::
