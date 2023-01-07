from collections import Counter
from sklearn.metrics import roc_curve, auc
import matplotlib.pyplot as plt

import numpy as np
import re
import pandas as pd

import itertools
from tqdm import tqdm
from tqdm._tqdm_notebook import tqdm_notebook
tqdm_notebook.pandas(desc='apply')

# utils
def strsplit(string, pattern=' '):
    '''
    split a string into a list of strings, according to some given pattern.
    '''
    return(re.split(pattern, str(string)))

def get_max_index(l,n):
    '''
    get the indices corresponding to the n largest entries in a given list
    '''
    return list(pd.Series(l).sort_values(ascending=False).index[0:n])

def in_or_not(l,s):
    '''
    check if a string s is in another string l
    '''
    return(s in l)

def get_idf(w,pos,neg):
    '''
    compute the measure "DF"(difference in frequency), as proposed in the report file
    '''
    return np.abs(np.sum(pos.apply(in_or_not,s=w))-
                 np.sum(neg.apply(in_or_not,s=w)))/max(len(pos),len(neg))

def get_acc(prob,true,thr):
    '''
    compute the overall accuracy of prediction
    Args:
        prob: list, a list of probability scores of class 1
        true: list, a list of true labels, taking the values of 0 or 1
        thr: float, a probability threshold in assigning predicted labels
        
    Returns: float, accuracy ranging from 0 to 1
    '''
    n1 = np.sum(np.multiply(np.array(prob)>=thr,np.array(true)==1))
    n2 = np.sum(np.multiply(np.array(prob)<thr,np.array(true)==0))
    return (n1+n2)/len(prob)


class naive_bayes(object):
    '''
    This class integrates all functions involved in a Naive Bayes classifier,
     including construction of vocabulary, parameter estimation, prediction and model evaluation. 
    
    '''
    def __init__(self, min_freq, vocab_size):
        '''
        initialize the classifier
        Args: 
            min_freq: words that appear no more than min_freq times would be discarded
            vocab_size: number of words to include in the vocabulary. Note that we will keep
                d words with the greatest TFDF value for positive and negative tweets respectively,
                and then pool them together, with duplicated words dropped. 
        '''
        self.min_freq = min_freq
        self.vocab_size = vocab_size
        self.idf = np.zeros((0))
    def __get_vocab(self, pos, neg, min_freq):
        '''
        get the raw vocabulary
        Args：
            pos: a list of strings, each of which is a positive tweet
            neg: a list of strings, each of which is a negative tweet
            min_freq: minimum word frequency allowed
            
        Returns:
            pos_words_: a list of words that appear in positive tweets
            neg_words_: a list of words that appear in negative tweets
            vocab: a list of words that appear more than min_freq in all tweets
            pos_words: a list of lists, each of which is a list of words from a positive tweet
            neg_words: a list of lists, each of which is a list of words from a negative tweet
        '''
        # returns an overall vocabulary without repetition, and all the words that appear in positive and
        # negative instances (which belong to the vocabulary and with repetition)
        # pos and neg: positive and negative instances respectively, given in a list
        pos_words = pos.apply(func=strsplit)
        pos_words_ = [w for ws in pos_words for w in ws]
        neg_words = neg.apply(func=strsplit)
        neg_words_ = [w for ws in neg_words for w in ws]
        counts = pd.Series(dict(Counter(pos_words_ + neg_words_)))
        vocab = list(counts[counts>min_freq].index)
        return(pos_words_, neg_words_, vocab, pos_words, neg_words)
    
    def __tf_idf(self, pos, neg, pos_, neg_, vocab):
        '''
        calculate TF and DF measures for positive and negative tweets respectively. 
        '''
        count_pos,count_neg = Counter(pos_),Counter(neg_)
        tf_pos = np.array([count_pos[w]/len(pos_) for w in vocab])
        tf_neg = np.array([count_neg[w]/len(neg_) for w in vocab])
        #idf = [np.abs(np.sum(pos.apply(inornot,s=w))-
        #             np.sum(neg.apply(inornot,s=w)))/max(len(pos),len(neg)) for w in tqdm(vocab)]
        print('Extracting key words...')
        idf = np.array(pd.Series(vocab).progress_apply(get_idf, pos=pos, neg=neg))
        self.tf_pos, self.tf_neg, self.idf = tf_pos, tf_neg, idf
    
    def __get_key_vocab(self, num):
        '''
        get the vocabulary of the Naive Bayes classifier.
        a number of "num" words with the largest TFDF values are subtracted, for positive and negative tweets respectively.   
        '''
        self.key_vocab = [self.vocab[i] for i in set(get_max_index(self.tf_pos*self.idf,n=num) + get_max_index(self.tf_neg*self.idf,n=num))]
    
    def __get_prob(self, key_vocab, pos_, neg_):
        '''
        get the frequency of a word in positive or negative tweets. Laplace smoothing is used to avoid zero probability. 
        '''
        count_pos,count_neg = Counter(pos_),Counter(neg_)
        prob_pos = [(count_pos[w]+1)/(len(pos_)+len(key_vocab)) for w in key_vocab]
        prob_neg = [(count_neg[w]+1)/(len(neg_)+len(key_vocab)) for w in key_vocab]
        return(prob_pos, prob_neg)
    
    def __get_pi(self, pos, neg):
        '''
        get the prior probability of positive and negative examples, estimated by their frequencies
        '''
        total = len(pos)+len(neg)
        return(len(pos)/total, len(neg)/total)
    
    def __predict(self, pi_pos, pi_neg, prob_pos, prob_neg, key_vocab, text):
        '''
        predicts the posterior probability that a piece of text contains positive sentiment 
        '''
        # text should be a list of words
        text = [w for w in text if w in key_vocab]
        p1 = np.prod(np.array([prob_pos[key_vocab.index(w)] for w in text]))*pi_pos
        p0 = np.prod(np.array([prob_neg[key_vocab.index(w)] for w in text]))*pi_neg
        predict_pos = p1/(p1+p0)
        return(predict_pos,1-predict_pos)
    
    def train(self, pos, neg):
        '''
        trains the classifier (i.e. parameter estimation)
        '''
        if len(self.idf)==0:
            # since the construction of vocabulary is time consuming, 
            ##  the following command will be implemented only when self.idf is not yet calculated.
            self.pos_words_, self.neg_words_, self.vocab, self.pos_words, self.neg_words = self.__get_vocab(pos, neg, min_freq=self.min_freq)
        #self.key_vocab = self.__tf_idf(self.pos_words,self.neg_words,self.pos_words_,self.neg_words_,self.vocab,self.vocab_size)
            self.__tf_idf(self.pos_words,self.neg_words,self.pos_words_,self.neg_words_,self.vocab)
        self.__get_key_vocab(self.vocab_size)
        self.prob_pos, self.prob_neg = self.__get_prob(self.key_vocab, self.pos_words_, self.neg_words_)
        self.pi_pos, self.pi_neg = self.__get_pi(self.pos_words, self.neg_words)
        
    def predict(self, text, processor):
        '''
        first process a piece of text with a given processor, then predict the probability that it contains positive sentiment
        '''
        prob = self.__predict(self.pi_pos, self.pi_neg, self.prob_pos, self.prob_neg, self.key_vocab, 
                       processor.process(str(text)).split(' '))
        return(prob[0])

    def evaluate(self, test, processor, roc=True, path=None, prob=False):
        '''
        evaluate the classifier on a given dataset, and return various evaluation metrics, like ROC curve, AUC, recall and accuracy.
        Note: The given test set must be a pandas dataframe with columns named "text" and "label".
        '''
        if roc==False and path!=None:
            # raise a warning message, when ROC curve is not required, but a figure path is specified
            warning_message = "ROC Curve is not required but a figure path is given. The path is thereby omitted. "
            warnings.warn(warning_message)
        print('Text processing and predicting...')
        prob = list(test['text'].progress_apply(self.predict, processor=processor))
        true = [1 if test['label'][i]==4 else 0 for i in range(len(test))]
        
        fpr, tpr, thr = roc_curve(true, prob)
        best_index = get_max_index(tpr-fpr,1) # get the optimal threshold by maximizing Youden Index
        best_fpr, best_tpr, best_thr = fpr[best_index], tpr[best_index], thr[best_index]
        roc_auc = auc(fpr, tpr)
        acc = get_acc(prob, true, best_thr)
        if roc: # if ROC curve is required
            plt.figure(figsize=(5,5), dpi=100)
            plt.plot(fpr, tpr, color='darkorange', lw=2, label='ROC curve (AUC = %0.2f)' % roc_auc)
            plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
            plt.scatter(best_fpr, best_tpr, color='red', marker='H', 
                        label='optimal youden index\nthreshold=%.2f\nfalse positive rate=%.2f\ntrue positive rate=%.2f'%(best_thr[0],best_fpr[0],best_tpr[0]))
            plt.xlim([0.0, 1.0])
            plt.ylim([0.0, 1.05])
            plt.xlabel('False Positive Rate', fontsize=20)
            plt.ylabel('True Positive Rate', fontsize=20)
            plt.title('ROC curve', fontsize=20)
            plt.legend(loc="lower right", fontsize=10)
            if path:
                plt.savefig(path)
            plt.show()
        # save the evaluation results in a dictionary
        results = {'accuracy':acc, 'AUC':roc_auc, 'best threshold':best_thr,
                  'recall of positive examples':best_tpr, 'recall of negative examples':1-best_fpr}
        if prob: # if predicted probabilities of test set are required
            results['prob']=prob
            results['true']=true
        return(results)