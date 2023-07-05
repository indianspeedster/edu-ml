
# Methodology

## Train 3 different models
1. On the entire dataset
2. Full few shot (10 samples of each class)
3. Full few shot (10 samples of each class) on augmented dataset
   - Augmentation strategy: Synonym replacement.
   

# Algorithm

## Entire dataset
- Training a Bert classifier with the Trainer object of the transformer.

## Full few shot
1. Create a dataset object of 10 samples of each intent.
2. Train using a 3-fold cv (3 as the minimum number of samples in intent is 35).
3. For training, follow the same process as done in the Entire dataset.

## Augmentation using SR (Synonym Replacement)
- Replace "n" non-stop words randomly using the formula n = alpha * length. Length should be the length of non-stop words.
- Apply the above function to the speech_text.
- Create a dataset object of 20 samples of each intent where 10 are the original intent and 10 are the augmented version.
- Train using a 3-fold cv.


