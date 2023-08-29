# Impact of data augmentation on intent classification using full few-shot learning.
## Abstract: 
This research paper uses a powerful technique called full few-shot learning to explore how data augmentation affects intent classification. Intent classification is about understanding the purpose of a sentence or text. With full few-shot learning, we can make the model work even when we have very little training data. In our experiments, we used a BERT Large model and only ten examples for each intent. Then, we tried adding more samples through data augmentation by using the synonym replacement strategy to see if it made the model better. After training and evaluation,  we found that there was only a minor improvement in accuracy and hence we came to the conclusion that data augmentation using synonym replacement is not impactful in the case of full few-shot learning.

## Introduction:
A significant challenge in building an intent classifier is the collection and labeling of training data. It requires a considerable amount of manpower and surveys to gather data and label how people communicate with conversational agents for a specific task. In today's era, one of the robust techniques for solving the issue of data scarcity is data augmentation, which has proven to be powerful in tasks involving vision and speech data. However, when dealing with language data, things become different as devising universal data augmentation techniques becomes highly challenging.

Despite data augmentation for language data being a demanding task, researchers have developed various techniques to augment natural language data, which have shown some degree of success. In a well-known study [2], the concept of translating data into a different language and then back-translating it to the original language was proposed. Another study focused on the idea of fine-tuning large language models for rephrasing the data. Although these studies have demonstrated success in certain tasks, implementing them can be costly for specific purposes. A cost-effective approach to language data augmentation was published by [], representing the first comprehensive exploration of text editing techniques for this purpose. This work proposed 4 different ways to augment language data, (Synonym replacement, Random insertion, Random swap, and Random deletion). We found the idea of synonym replacement to be a strong subject for our experiment. The reason was that in terms of communication with conversational agents, there is a high chance that different people use different words for the same intent due to which we expected that synonym replacement would help in our experiment.

Our experiment primarily focus on full few shot Intent classification on HWU 64 dataset, We primarily focused on 3 scenarios about which we will be discussing further in the methodology section. 

## Methodology:
We decided to train an Intent classifier, where the intent is a type of request that a conversational agent supports e.g: the user can ask the agent to set an Alarm, play music, etc. To understand in-depth and compare results we considered training 3 different Intent classifiers.

- Classifier built on Full data.
- Classifier build on full few-shot data. 
- Classifier built on full few-shot data and augmented data.

One of the important tasks in the process was to decide on an augmentation strategy, Augmentation can be not that difficult when it is done on Image data but when it comes to language data, augmentation can be a daunting task where you have to make sure that the context in the language remains the same.
In our case, we followed **Synonym Replacement**, The idea was to replace the words in the speech text with their synonyms but the issue was if we replace all the words with their synonyms then there are high chances that the context won't remain the same. so to avoid this we came up with the idea of randomly choosing n words from the sentence that are not stop words and replacing each of these words with one of its synonyms chosen at random.
But the next issue was that long sentences have more words than short ones, they can absorb more noise while maintaining their original class label. To counter this we went ahead and followed a synonym replacement approach mentioned in []. The approach focuses on making n directly proportional to the length of the word and calculating n with the help of a constant "α" and the size of the sentence l. This led us to the formulae n = α * l and the value of "α" lies between 0 and 1.
But the next issue was that long sentences have more words than short ones, they can absorb more noise while maintaining their original class label. To counter this we went ahead and followed a synonym replacement approach mentioned in []. The approach focuses on making n directly proportional to the length of the word and calculating n with the help of a constant "α" and the size of the sentence l. This led us to the formulae n = α * l and the value of "α" lies between 0 and 1.
This made sure that only a specific ratio of the total words are getting replaced and hence led to high chances of overall context being maintained.

## Datasets:

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

The source of the data is the official github repository of HWU64 dataset.

### Training and Evaluation
For training, we used BERT(Bidirectional Encoder Representation for Transformers) large model. BERT is a pre-trained language model that can understand the context of words in a sentence by considering the words that come before and after each word. This bidirectional approach allows BERT to capture complex language patterns and meanings, making it highly effective for a wide range of NLP tasks, such as text classification, sentiment analysis, question answering, and more. For our use case, we added a linear classification layer on top of the [CLS] token. We trained in 3 different setups and we selected a validation set to avoid issues with unstable hyperparameter tunning and focus on assessing the quality of the generated data. We will discuss more about the setup in the next part.

**1. Training on Entire Dataset:** In this configuration, we undertook training the BERT Large uncased model using the complete dataset. Hyperparameters were tuned, including learning rate, batch size, and number of training epochs. The choice of learning rate influenced the convergence speed of the optimization process, while the batch size of 8 played a role in memory utilization during training. The number of training epochs was set as 10 to prevent overfitting while ensuring optimal learning.

**2. Full few shot setup:** In this second configuration we randomly picked 30 samples from each of the 64 labels and then made a 3-fold data of 10 samples each. Next, we took the same BERT large uncased model to train on this full few-shot dataset. 

**3. Full few shot and augmented setup:** In this configuration we picked the same 30 samples and with the help of the data augmentation strategy described above we added 30 more data and now the total sample count is 60. Then we divided the 60 samples of 64 intent into 20 samples each and made three fold dataset. Next again we took the same Bert Large uncased model to train this dataset.

## Results

Our experiments on the above three setups provided us with the results mentioned in Table 1. 

When we trained the Bert large uncased model using all the samples provided in the entire dataset, it performed really well, with an accuracy of 92.75%. This accuracy was not a surprise as BERT-large (Devlin et al., 2018) is a massive model with 340 million parameters. However, when we trained it with only a few examples for each intent (the full few-shot dataset), the accuracy dropped to 83%. Although it's difficult to say that 83% accuracy is good accuracy or bad accuracy considering that the model was trained on a classification problem on 64 labels.

Adding more examples through our data augmentation strategy the accuracy moved up by a small percentage and the result was 84.5%. Even though there was some improvement but this improvement was not impactful as the results were not that close to the full dataset model.

<div align="center">
  <table>
    <tr>
      <th>Model</th>
      <th>Accuracy</th>
    </tr>
    <tr>
      <td>Full dataset</td>
      <td>92.75</td>
    </tr>
    <tr>
      <td>Full few-shot dataset</td>
      <td>83</td>
    </tr>
    <tr>
      <td>Full few-shot dataset + Augmented dataset</td>
      <td>84.5</td>
    </tr>
  </table>
  </div>

## Discussion
Our study's results have opened up an interesting discussion about why adding more examples through data augmentation didn't lead to a significant increase in accuracy. Let's explore some possible reasons.

One key factor is how we changed the words in our augmentation technique. We used synonyms to make new examples, but these changes might not have been big enough to make the model understand better. Using more advanced methods to change sentences, like rephrasing them in different ways, might lead to more noticeable improvements.

Another reason could be that intent classification is tricky, especially when we have very few examples. Even though we tried to help the model by adding more examples, the limited number of training instances might have made it hard for the model to understand all the different ways people express their intentions.

Also, the dataset we used is quite diverse, covering lots of different topics and situations. This diversity might have made it tough for our augmentation strategy to create truly meaningful variations for each intent.

To sum up, our study gives us valuable hints about how complex intent classification can be. While data augmentation is promising, it might not work perfectly in all cases. Exploring more advanced augmentation methods and larger datasets could help us get better results in the future.

## References

[1] Sahu, Gaurav, et al. "Data augmentation for intent classification with off-the-shelf large language models."

[2] Yu, Adams Wei, et al. "Qanet: Combining local convolution with global self-attention for reading comprehension."

[3] Liu, Xingkun, et al. "Benchmarking natural language understanding services for building conversational agents."

[4] Wei, Jason, and Kai Zou. "Eda: Easy data augmentation techniques for boosting performance on text classification tasks."

[5] Devlin, Jacob, et al. "Bert: Pre-training of deep bidirectional transformers for language understanding."
