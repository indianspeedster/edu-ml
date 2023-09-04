# Impact of data augmentation on intent classification using Few shot learning.
## Abstract: 
This research paper uses a powerful technique called Few shot learning to explore how data augmentation affects intent classification. Intent classification is about understanding the purpose of a sentence or text. With Few shot learning, we can make the model work even when we have very little training data. In our experiments, we used a BERT Large model and only ten examples for each intent. Then, we tried adding more samples through data augmentation by using the synonym replacement strategy to see if it made the model better. After training and evaluation,  we found that there was only a minor improvement in accuracy and hence we came to the conclusion that data augmentation using synonym replacement is not impactful in the case of Few shot learning.

## Introduction:
A significant challenge in building an intent classifier is the collection and labeling of training data[1]. It requires a considerable amount of manpower and surveys to gather data and label how people communicate with conversational agents for a specific task. Also intent classification is often used on voice assistants and there comes a scenario when users may want to add their own custom intents but they end up having only a few example of new intent or an issue of data scarcity. Today's era, one of the robust techniques for solving the issue of data scarcity is data augmentation, which has proven to be powerful in tasks involving vision and speech data. However, when dealing with language data, things become different as devising universal data augmentation techniques becomes highly challenging.

Despite data augmentation for language data being a demanding task, researchers have developed various techniques to augment natural language data, which have shown some degree of success. In a well-known study [2], the concept of translating data into a different language and then back-translating it to the original language was proposed. Another study focused on the idea of fine-tuning large language models for rephrasing the data. Although these studies have demonstrated success in certain tasks, implementing them can be costly for specific purposes. A cost-effective approach to language data augmentation was published by [4], representing the first comprehensive exploration of text editing techniques for this purpose. This work proposed 4 different ways to augment language data, (Synonym replacement, Random insertion, Random swap, and Random deletion). We found the idea of synonym replacement to be a strong subject for our experiment. The reason was that in terms of communication with conversational agents, there is a high chance that different people use different words for the same intent due to which we expected that synonym replacement would help in our experiment.

Prior work [7] has shown that attention mechanism has helped a lot in Natural language understanding task. Also previous work [6] claims that training Intent classifier using BERT [5] has achieved significant improvement on intent classification accuracy . Considering which an optimal choice to train the intent classification model for this task was BERT.

Our experiment primarily focus on Few shot Intent classification on HWU 64 dataset [3], We primarily focused on 3 scenarios about which we will be discussing further in the methodology section. 

## Methodology:
We decided to train an Intent classifier, where the intent is a type of request that a conversational agent supports e.g: the user can ask the agent to set an Alarm, play music, etc. To understand in-depth and compare results we considered training 3 different Intent classifiers.



- Classifier built on Full data.
- Classifier build on Few shot data. 
- Classifier built on Few shot data and augmented data.

One of the important tasks in the process was to decide on an augmentation strategy, Augmentation can be not that difficult when it is done on Image data but when it comes to language data, augmentation can be a daunting task where you have to make sure that the context in the language remains the same.
In our case, we followed **Synonym Replacement**, The idea was to replace the words in the speech text with their synonyms but the issue was if we replace all the words with their synonyms then there are high chances that the context won't remain the same. So to avoid this we followed the approach mentioned in [8]. It focuses on randomly choosing n words from the sentence that are not stop words and replacing each of these words with one of its synonyms chosen at random.
But the next issue was that long sentences have more words than short ones, they can absorb more noise while maintaining their original class label. To counter this we went ahead and followed a synonym replacement approach mentioned in [4]. The approach focuses on making n directly proportional to the length of the word and calculating n with the help of a constant "α" and the size of the sentence l. This led us to the formulae n = α * l and the value of "α" lies between 0 and 1.
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

The source of the data is the official github repository https://github.com/xliuhw/NLU-Evaluation-Data

### Data Augmentation
For data augmentation, we followed an augmentation strategy named Synonym Replacement as mentioned in [4]. The idea was to randomly choose n words from the sentence that are not stop words. Replace each of these words with one of its synonyms chosen at random. Since long sentences have more words than short ones, they can absorb more noise while maintaining their original class label. To compensate the authors vary the number of words changed, n, based on the sentence length l with the formula n=αl, where α is a parameter that indicates the percent of the words in a sentence are changed.

### Scenarios
We trained in 3 different setups and we selected a validation set to avoid issues with unstable hyperparameter tunning and focus on assessing the quality of the generated data. We will discuss more about the setup in the next part.

**1. Full Dataset setup:** In this scenario we used the entire train set specified above in the dataset section, The total count was 8954 samples of 64 different intent for training.

**2. Few shot setup:** In this second scenario we randomly picked 30 samples from each of the 64 labels and then made a 3-fold data of 10 samples each of 64 different intent. 

**3. Few shot and augmented setup:** In this scenario we picked the same 30 samples and with the help of the data augmentation strategy described above we added 30 more data and now the total sample count is 60. This resulted in 3 fold data of 10 few shot data and 10 augmented versions of the same


### Training and Evaluation
For training, we used BERT(Bidirectional Encoder Representation for Transformers) large model [3]. BERT is a pre-trained language model that can understand the context of words in a sentence by considering the words that come before and after each word. This bidirectional approach allows BERT to capture complex language patterns and meanings, making it highly effective for a wide range of NLP tasks, such as text classification, sentiment analysis, question answering, and more. For our use case, we added a linear classification layer on top of the classifier token. 

## Results

Our experiments on the above three setups provided us with the results mentioned in Table 1. 

When we trained the Bert large uncased model using all the samples provided in the entire dataset, it performed as expected producing an accuracy of 92.75%. This accuracy was not a surprise as BERT-large [5] is a massive model with 340 million parameters. However, when we trained it with only a few examples for each intent (the Few shot dataset), the accuracy dropped to 83%. Although it's difficult to say that 83% accuracy is good accuracy or bad accuracy considering that the model was trained on a classification problem on 64 labels.

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

<div align="center">
  <table>
    <tr>
      <th>Original</th>
      <th>Augmented</th>
      <th>Intent</th>
    </tr>
    <tr>
      <td>book a taxi for tomorrow morning</td>
      <td>book a hack for tomorrow morning</td>
      <td>transport_taxi</td>
    </tr>
    <tr>
      <td>what groups are listed in my contacts</td>
      <td>what groups are name in my contacts</td>
      <td>lists_query</td>
    </tr>
    <tr>
      <td>tell me what's happening this week</td>
      <td>tell me what's happening this hebdomad</td>
      <td>calendar_query</td>
    </tr>
  </table>
  </div>

Above are some of the examples of the original data and augmented version of the same, When we closely look at the augmented version we see that although the replacement words are synonyms but it's very rare that people use these words to express their intention. For eg, when we see the replacement word for "taxi" is "hack". Although hack is a synonym of taxi as mentioned in Google dictionary but people mostly use hack in terms of tech related subjects. Similarly we can see that "week" is replaced by the word "hebdomad" but again same issue here too, very rare that someone uses hebdomad to refer week.

Another key factor is how we changed the words in our augmentation technique. We used synonyms to make new examples, but these changes might not have been big enough to make the model understand better. Using more advanced methods as mentioned in [1] to change sentences, like rephrasing them using large lnguage models, might lead to more noticeable improvements.

To sum up, our study gives us valuable hints about how complex intent classification can be. While data augmentation using synonym replacement is promising, it might not work perfectly in all cases. Exploring more advanced augmentation methods and larger datasets could help us get better results in the future.

## References

[1] Sahu, Gaurav, et al. "Data augmentation for intent classification with off-the-shelf large language models."

[2] Yu, Adams Wei, et al. "Qanet: Combining local convolution with global self-attention for reading comprehension."

[3] Liu, Xingkun, et al. "Benchmarking natural language understanding services for building conversational agents."

[4] Wei, Jason, and Kai Zou. "Eda: Easy data augmentation techniques for boosting performance on text classification tasks."

[5] Devlin, Jacob, et al. "Bert: Pre-training of deep bidirectional transformers for language understanding."

[6] Chen, Qian, Zhu Zhuo, and Wen Wang. "Bert for joint intent classification and slot filling." arXiv preprint arXiv:1902.10909 (2019).

[7] Dzmitry Bahdanau, Kyunghyun Cho, and Yoshua Bengio. 2014. Neural machine translation by jointly learning to align and translate. CoRR, abs/1409.0473.

[8] Bonthu, Sridevi, et al. "Effective text augmentation strategy for nlp models." Proceedings of Third International Conference on Sustainable Computing: SUSCOM 2021. Springer Singapore, 2022
