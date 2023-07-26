# Ambiguous Details
1. Choosing between two datasets (26k and 10k)
2. Creating the labels (picking the intent from the data directly or merging scenarios and intent to build a unique intent)
3. While calculating n = α * (length of sentence) (keeping stopwords in total count or leaving stop words in total count).
4. Picking the upper or lower bound while calculating n.
5. Skipping the word when no synonyms are found or picking a different nonstopword and replacing it.
6. Using same validation set for all 3 models or creating a separate one for each on the basis of the total data that the model is using for training 
7. Choice between using Adamw and SGD as optimizers.
8. Hyperparameter search on the largest model or on all 3 models.
