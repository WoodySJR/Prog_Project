## 1. Project introduction

In this project, we focus on the sentiment analysis of Tweets (i.e. messages posted on Twitter), and analyze how text pre-processing techniques and different machine learning algorithms affect the accuracy of sentiment analysis. To be more specific, in this work we focus on two major text pre-processing techniques namely "lemmatization" and "stopword deletion", and three machine learning algorithms namely Naive Bayes, LSTM and BERT. 

What's noteworthy is that we propose a novel measure called NFDF to identify important words, in order to construct the vocabulary for a Naive Bayes classifier, and write a Python Class in which Naive Bayes is implemented from scratch(stored in __"naive_bayes.py"__). Besides, for the sake of convenience, we integrate all our text pre-processing functions into a Python Class(stored in __"preprocess.py"__).

In this work, we come to the following conclusions:
(a) Lemmatization only helps when the vocabulary of Naive Bayes is sufficiently small, probably because it spares more space for words with different meanings. As the vocabulary size gets bigger, lemmatization begins to harm model performance. For LSTM and BERT, lemmatization never helps, probably because both of them take raw sequences of words as input and thereby are capable of learning sentence syntax, while lemmatization breaks syntax. 
(b) The deletion of stopwords never benefits model performance, and this indicates that stopwords might play an important role in the sentiment analysis of casually written texts(eg. tweets). But this still needs to be verified by checking the effect of stopword deletion in formal texts like news. 
(c) BERT model with one fully-connected output layer and without lemmatization or stopword deletion achieves the best predicting performance on validation set. This model eventually achieves an overall accuracy of 83.6% on test data, with recalls of positive and negative tweets equal to 80.2% and 87.0% respectively.

We provide all our codes and data for your further research, and our results are replicable because we set a random seed 2022 in model initialization.  Our research report is also provided in __"report.pdf"__ for your reference. If you have any suggestions or new ideas, please feel free to contact us through: woodysjr@foxmail.com. 

## 2. Data Description

In this work, we utilize the dataset called "Sentiment140" available at: http://help.sentiment140.com/for-students/, which started as a class project from Starford University in 2009. This dataset was collected through Twitter Search API by Go et al.(2009)[1]. The training set of size 1.6 million was automatically annotated as positive or negative, depending on whether the tweets include positive or negative emoticons, such as :) and :(; The test set of size 498 was manually labelled, and we discard those samples that were annotated as "neutral". Due to the limitation of computational resources, we only keep 20,000 positive and 20,000 negative comments from the training set for our analysis, and further extract 20% to be our validation set for hyperparameter tuning and model selection. We finally evaluate the predicting accuracy of our model on the test set. 

For the sake of convenience, we provide the original test set in __"test.csv"__, together with our processed version of training data in __"40k_split_processed.csv"__. The original training set with 1,600,000 observations is too big to upload. Please visit http://help.sentiment140.com/for-students/ to download. Note that in our processed version, only 40,000 observations are kept, as mentioned above, and we add 5 more columns: 
(1) "status": takes the value of "t" or "v", indicating whether the observation is in training set or validation set.
(2) "none": texts pre-processed with basic techniques, like converting to lower forms, deleting digits and punctuations, and converting "@username" and weblinks into special tokens "USERNAME" and "URL" respectively. 
(3) "lemma": texts pre-processed with lemmatization, in addition to basic techniques mentioned above.
(4) "delstop": texts pre-processed with stopword deletion, in addition to basic techniques mentioned above.
(5) "lemma&delstop": texts pre-processed with both lemmatization and stopword deletion.

## 3. Contents
Our project can be roughly divided into seven parts, as in __"main.ipynb"__.

__Part 1:__  data load-in; initialize text processors and pre-process all tweets. \
__Part 2:__  An exploratory analysis of tweets. Separate wordclouds of positive and negative tweets are plotted to get a first glimpse of what they look like. A customized function for generating wordclouds is utilized here, and stored in "custom_wordcloud.py".\
__Part 3:__  Naive Bayes classification and hyperparameter tuning.\
__Part 4:__  LSTM classification and hyperparameter tuning.\
__Part 5:__  BERT classification and hyperparameter tuning.\
__Part 6:__  grid search results are visualized to find the optimal model.\
__Part 7:__ re-train our best model on the whole training set, and evaluate it on test set. 
    
Note: functions needed in neural network training and evaluation are all stored in __"utilities.py"__. 

## 4. Environment

To ensure replicability, our environment and package versions are listed below. 
      
      Python 3.8.8;
      d2lzh 1.0.0;
      ipykernel 5.3.4;
      jupyter 1.0.0;
      matplotlib 3.6.1;
      mxnet 1.9.1;
      nltk 3.6.1;
      numpy 1.22.4;
      pandas 1.5.1;
      scikit-learn 0.24.1;
      tensorboard 2.11.0;
      tqdm 4.64.1.
      
To replicate our results, please follow the detailed comments in "main.pynb". 

## References:
[1] Go, A., Bhayani, R., and Huang, L. (2009). Twitter sentiment classification using distant supervision.
CS224N project report, Stanford, 1(12):2009.
