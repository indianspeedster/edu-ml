# Impact of data augmentation on intent classification using full few-shot learning.
## Abstract: 
This research paper uses a powerful technique called full few-shot learning to explore how data augmentation affects intent classification. Intent classification is about understanding the purpose of a sentence or text. With full few-shot learning, we can make the model work even when we have very little training data. In our experiments, we used a BERT Large model and only ten examples for each intent. Then, we tried adding more samples through data augmentation by using the synonym replacement strategy to see if it made the model better. After training and evaluation,  we found that there was only a minor improvement in accuracy and hence we came to the conclusion that data augmentation using synonym replacement is not impactful in the case of full few-shot learning.
## Methodology:
We decided to train an Intent classifier, where the intent is a type of request that a conversational agent supports eg: the user can ask the agent to set an Alarm, play music, etc. To understand in-depth and compare results we considered training 3 different Intent classifiers.

- Classifier built on Full data.
- Classifier build on full few-shot data. 
- Classifier built on full few-shot data and augmented data.

One of the important tasks in the process was to decide on an augmentation strategy, Augmentation can be not that difficult when it is done on Image data but when it comes to language data, augmentation can be a daunting task where you have to make sure that the context in the language remains the same.
In our case, we followed **Synonym Replacement**, The idea was to replace the words in the speech text with their synonyms but the issue was if we replace all the words with their synonyms then there are high chances that the context won't remain the same. so to avoid this we came up with the idea of randomly choosing n words from the sentence that are not stop words and replacing each of these words with one of its synonyms chosen at random.
But the next issue was that long sentences have more words than short ones, they can absorb more noise while maintaining their original class label. To decide this we came up with the idea of making n directly proportional to the length of the word and calculating n with the help of a constant "α" and the size of the sentence l. This led us to the formulae n = α * l and the value of "α" lies between 0 and 1.
This made sure that only a specific ratio of the total words are getting replaced and hence led to high chances of overall context being maintained.
### Datasets:

For our intent classification model, we used the HWU64 dataset which is a multi-domain dataset each covering a wide range of typical task-oriented chatbot domains, such as setting up a calendar, adding remainder or alarm, etc. In the table below we have described the data and how exactly we have used the data to train test and validate our model.
<div align="center">
  <table>
    <tr>
      <th></th>
      <th>Count</th>
    </tr>
    <tr>
      <td>domains</td>
      <td>18</td>
    </tr>
    <tr>
      <td>Intents</td>
      <td>64</td>
    </tr>
    <tr>
      <td>train samples</td>
      <td>8954</td>
    </tr>
    <tr>
      <td>val samples</td>
      <td>1076</td>
    </tr>
    <tr>
      <td>test samples</td>
      <td>1076</td>
    </tr>
  </table>
</div>

### Training and Evaluation
For training, we used BERT(Bidirectional Encoder Representation for Transformers) large model. BERT is a pre-trained language model that can understand the context of words in a sentence by considering the words that come before and after each word. This bidirectional approach allows BERT to capture complex language patterns and meanings, making it highly effective for a wide range of NLP tasks, such as text classification, sentiment analysis, question answering, and more. For our use case, we added a linear classification layer on top of the [CLS] token. We trained in 3 different setups and we selected a validation set to avoid issues with unstable hyperparameter tunning and focus on assessing the quality of the generated data. We will discuss more about the setup in the next part.

**1. Training on Entire Dataset:** In this configuration, we undertook training the BERT Large uncased model using the complete dataset. Hyperparameters were tuned, including learning rate, batch size, and number of training epochs. The choice of learning rate influenced the convergence speed of the optimization process, while the batch size of 8 played a role in memory utilization during training. The number of training epochs was set as 10 to prevent overfitting while ensuring optimal learning.

**2. Full few shot setup:** In this second configuration we randomly picked 30 samples from each of the 64 labels and then made a 3-fold data of 10 samples each. Next, we took the same BERT large uncased model to train on this full few-shot dataset. 

**3. Full few shot and augmented setup:** In this configuration we picked the same 30 samples and with the help of the data augmentation strategy described above we added 30 more data and now the total sample count is 60. Then we divided the 60 samples of 64 intent into 20 samples each and made three fold dataset. Next again we took the same Bert Large uncased model to train this dataset.

