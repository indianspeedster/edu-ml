# Ambiguous Details
1. Choosing between two datasets (26k and 10k)
2. Creating the labels (picking the intent from the data directly or merging scenarios and intent to build a unique intent)
3. While calculating n = α * (length of sentence) (keeping stopwords in total count or leaving stop words in total count).
4. Picking upper or lower bound while calculating n.
5. Skipping the word when no synonyms are found or picking different non stopword and replace it.
6. Choice between using Adamw and SGD as optimizers.
7. Hyperparameter search on the largest model or on all 3 models.
