Hey,

I am Shekhar and I am one of several students who are working on the project Using Reproducibility in Machine Learning Education under the mentorship of Fraida Fund. My Proposal aims to develop interactive educational materials about reproducibility in machine learning, for use in graduate and undergraduate classes

My work is inspired by my participation in the machine learning reproducibility challenge, where I discovered that many ML research papers seem to have complete methodology details initially. However, when attempting to reproduce the results, I found missing implementation details that could potentially lead to different outcomes than expected. This absence of specific information presents us with various choices to make. To address this issue, I believe it's essential to explore different options available to complete the methodology description while carefully considering the impact of each choice on the final results. As a part of my efforts to educate researchers and students, I created a fictitious research paper that focuses on identifying and addressing general missing methodology details. The paper highlights various choices that can be made to fill in the gaps and explores how these choices can impact the overall results of the research.

In our fictitious research paper, we wanted to see how data augmentation could help in full few-shot learning, a challenging task in machine learning. So, we used a powerful language model called BERT Large to classify different intents in the hwu64 dataset. Surprisingly, even with just ten examples for each intent, the model achieved an impressive accuracy of 84%. This shows that BERT Large has the amazing ability to learn from very little data and recognize new things it has never seen before. This quality could be really useful in real-life situations where we don't have a lot of labeled data to work with.

After seeing good results initially, we wanted to see how data augmentation could make the model even better. Data augmentation is a technique that creates more varied examples for training, making the model more reliable and less likely to memorize the data. We wanted to fill in the missing information about the data augmentation we used, so we came up with a smart plan. We decided to replace some words with their similar meaning words to create new training examples and make the model more versatile. This way, we hoped to see how data augmentation would affect the model's performance. The results led to an improvement in the model's accuracy from 84% to 85%. This may seem like a small increase, but in few-shot learning where there's not much training data, even a little improvement is significant. 

In the fictitious paper, we left some of the methodology details ambiguous and made a decision tree where each node represents ambiguous methodology details and branches out to different choices that are made at that instance. The leaf nodes will represent the final results, providing insights into the magnitude of the differences resulting from each node selection.

I  have achieved significant progress in our project, reaching some major milestones:

- Final Draft of Fictitious Paper: The final version of our made-up research paper is now ready. This paper covers all the ambiguous methodologies.

- Decision Tree with Ambiguous Details: We have successfully finalized the decision tree with intentionally ambiguous details from the paper.

- Notebooks Prepared: We have completed all the notebooks for the branch containing the path to the original implementation.

- Three Bert Large Models  Built: We have accomplished building three separate Bert Large models, each catered to intent classification under different scenarios. 






During the project, I faced both challenges and accomplishments that allowed me to explore the world of reproducibility. Few-shot learning, a new and exciting machine learning technique, was tricky because there wasn't much information available since researchers are still studying it. Making a data augmentation strategy for natural language data that would really work was hard too because it relies on large language models (LLMs), which we couldn't use in our introductory materials at this stage of the project. So, finding a way to enhance the task with data augmentation was a big challenge. Also, I had to be careful while writing the methodology, making sure it was a bit unclear but not too much. Thankfully, I managed to strike the right balance.


Despite the obstacles, I found the project exciting and gained valuable insights into few-shot learning and its applications. With support from my mentor and experience in the Machine Learning reproducibility challenge, I made progress and overcame challenges. Working on this project has been a valuable learning experience, and I am looking forward to continuing this work throughout the summer at Summer of Reproducibility.
