# Impact of data augmentation on intent classification using full few-shot learning.
## Abstract: 
This research paper uses a powerful technique called full few-shot learning to explore how data augmentation affects intent classification. Intent classification is about understanding the purpose of a sentence or text. With full few-shot learning, we can make the model work even when we have very little training data. In our experiments, we used a BERT Large model and only ten examples for each intent. Then, we tried adding more examples through data augmentation to see if it makes the model better. While the impact of data augmentation was not huge, the small improvement it brought was still significant. The extra examples helped the model recognize new intents better. This study shows how data augmentation can be helpful in intent classification, even with limited training data

## Methodology:
We decided to train an Intent classifier, where the intent is a type of request that a conversational agent supports eg: the user can ask the agent to set an Alarm, play music, etc. To understand in-depth and compare results we considered training 3 different Intent classifiers.

- Classifier built on Full data.
- Classifier build on full few-shot data. (10 samples of each intent)
- Classifier built on full few-shot data and augmented data. (10 samples of each intent and an augmented version of the same)

One of the important tasks in the process was to decide on an augmentation strategy, Augmentation can be not that difficult when it is done on Image data but when it comes to language data, augmentation can be a daunting task where you have to make sure that the context in the language remains the same.
In our case, we followed **Synonym Replacement**, The idea was to replace the words in the speech text with their synonyms but the issue was if we replace all the words with their synonyms then there are high chances that the context won't remain the same. so to avoid this we came up with the idea of randomly choosing n words from the sentence that are not stop words and replacing each of these words with one of its synonyms chosen at random.
But the next issue was that long sentences have more words as compared to short ones, they can absorb more noise while maintaining their original class label. To decide this we came up with the idea of making n directly proportional to the length of the word and calculating n with the help of a constant "α" and the length of the sentence l. This led us to the formulae n = α * l and the value of "α" lies between 0 and 1.

This made sure that only a certain ratio of the total words are getting replaced and hence led to high chances of context being maintained.
### Datasets:

For our intent classification model, we used the HWU64 dataset which is a multi-domain dataset each covering a wide range of typical task-oriented chatbot domains, such as setting up a calendar, adding remainder or alarm, etc.
