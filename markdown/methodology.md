
# Methodology

## Train 3 different models
1. On the entire dataset
2. Full few shot (10 samples of each class)
3. Full few shot (10 samples of each class) on augmented dataset
   - Augmentation strategy: Synonym replacement.
   - Augmentation strategy: Word replacement by considering POS and cosine similar words.

# Algorithm

## Entire dataset
- Training a Bert classifier with the Trainer object of the transformer.

## Full few shot
1. Create a dataset object of 10 samples of each intent.
2. Train using 3-fold cv (3 as the minimum number of samples in an intent is 35).
3. For training, follow the same process as done in the Entire dataset.

## Augmentation using SR (Synonym Replacement)
- Replace "n" non-stop words randomly using the formula n = alpha * length. Length should be the length of non-stop words.
- Apply the above function to the speech_text, and the count of the total words would be doubled.
- Create a dataset object of 10 samples of each intent.
- Train using 6-fold cv (6 as the minimum number of samples in an intent is 70).

## Augmentation using Cosine similarity and POS
- Replace non-stop words with their cosine similar words with a probability higher than a given threshold and also make sure that the replacement words have the same POS as the original word.
- Apply the above function to the speech_text, and the count of the total words would be doubled.
- Create a dataset object of 10 samples of each intent.
- Train using 6-fold cv (6 as the minimum number of samples in an intent is 70).
