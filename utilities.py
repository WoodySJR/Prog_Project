from mxnet.contrib import text
from mxnet import nd, autograd
import collections
from mxnet.gluon import data as gdata,loss as gloss, utils as gutils, nn, rnn
import d2lzh as d2l
from mxnet import init, gluon
import mxnet as mx
import time
import numpy as np
from tqdm import tqdm
from sklearn.metrics import roc_curve, auc

def pad(x):
    '''
    pad sequences to a pre-specified length
    '''
    if len(x)>max_len:
        return x[:max_len]
    else:
        return x + [vocab.token_to_idx['<pad>']]*(max_len-len(x))
    
    
def get_features(technique):
    '''
    transform data into the format required by MXNET. 
    Args:
        technique: specified which text pre-processing technique to use, optional values
          include "none", "lemma&delstop", "lemma" and "delstop". 
    '''
    words = [str(st).split(' ') for st in data[technique]]
    words_ = [tk for st in words for tk in st]
    counter = collections.Counter(words_)
    vocab = text.vocab.Vocabulary(counter,min_freq=5,reserved_tokens=['<pad>'])
    words_idx = [vocab.to_indices(x) for x in words]
    global max_len
    max_len = max([len(words[i]) for i in range(len(words))])
    features = nd.array([pad(x) for x in words_idx])
    emb = text.embedding.create('glove',pretrained_file_name='glove.6B.300d.txt', vocabulary=vocab)
    labels = nd.array([1 if data['label'][i]==4 else 0 for i in range(len(data))])
    
    train_index = np.where((data['status']=='t'))[0]
    test_index = np.where((data['status']=='v'))[0]
    
    features_train = features[train_index]
    features_test = features[test_index]
    labels_train = labels[train_index]
    labels_test = labels[test_index]

    dataset_train = gdata.ArrayDataset(features_train,labels_train)
    dataset_test = gdata.ArrayDataset(features_test,labels_test)
    iter_train = gdata.DataLoader(dataset_train,256,shuffle=True) 
    iter_test = gdata.DataLoader(dataset_test,256,shuffle=False)
    
    return(iter_train, iter_test, emb, vocab)


def get_features_2(technique):
    words = [str(st).split(' ') for st in data[technique]]
    words_idx = [vocab[x] for x in words]
    global max_len
    max_len = max([len(words[i]) for i in range(len(words))])
    features = nd.array([pad(x) for x in words_idx])
    labels = nd.array([1 if data['label'][i]==4 else 0 for i in range(len(data))])
    
    train_index = np.where((data['status']=='t'))[0]
    test_index = np.where((data['status']=='v'))[0]
    
    features_train = features[train_index]
    features_test = features[test_index]
    labels_train = labels[train_index]
    labels_test = labels[test_index]

    dataset_train = gdata.ArrayDataset(features_train,labels_train)
    dataset_test = gdata.ArrayDataset(features_test,labels_test)
    iter_train = gdata.DataLoader(dataset_train,256,shuffle=True) 
    iter_test = gdata.DataLoader(dataset_test,256,shuffle=False)
    
    return(iter_train, iter_test)


def _get_batch(batch, ctx):
    '''Return features and labels on ctx.'''
    features, labels = batch
    if labels.dtype != features.dtype:
        labels = labels.astype(features.dtype)
    return (gutils.split_and_load(features, ctx),
            gutils.split_and_load(labels, ctx), features.shape[0])

def softmax(x):
    '''
    softmax activation function for calculating output probabilities
    '''
    return x[:,1].exp()/(x[:,0].exp()+x[:,1].exp())


def train(train_iter, test_iter, net, trainer, ctx, num_epochs):
    '''
    model training and evaluation
    Args:
        train_iter: training set in MXNET format
        test_iter: test set in MXNET format
        net: the model to be trained
        trainer: the optimizer to use
        ctx: the context in which the model resides
        num_epochs: number of epochs
       
    '''
    if isinstance(ctx, mx.Context):
        ctx = [ctx]
    for epoch in tqdm(range(num_epochs)):
        train_l_sum, n = 0.0, 0
        # train
        for i,batch in enumerate(train_iter):
            Xs, ys, batch_size = _get_batch(batch, ctx)
            ls = []
            with autograd.record():
                y_hats = [net(X) for X in Xs]
                # softmax-cross-entropy loss
                ls = [loss(y_hat,y) for y_hat,y in zip(y_hats,ys)][0]
            ls.backward()
            trainer.step(batch_size)
            train_l_sum += ls.sum().asscalar()
            n += batch_size
        
        # evaluate
        for i,batch in enumerate(iter_test):
            Xs,ys,batch_size = _get_batch(batch, d2l.try_all_gpus())
            if i == 0:
                y_pred = [net(X) for X in Xs][0]
                y_true = ys[0]
            else:
                y_pred = nd.concat(y_pred, [net(X) for X in Xs][0], dim=0)
                y_true = nd.concat(y_true, ys[0], dim=0)
        fpr, tpr, thr = roc_curve(y_true.asnumpy(), softmax(y_pred).asnumpy())
        roc_auc = auc(fpr, tpr)
        result = {'technique':t, 'num_hiddens':h, 'num_layers_lstm':l1, 'num_layers_ffn':l2,
                 'epoch':epoch, 'AUC':roc_auc}
        global results
        results = results.append(result, ignore_index=True)
        
def train_2(train_iter, test_iter, net, trainer, ctx, num_epochs):
    '''
    a modified version of training fuction that only prints out evaluation results.
    '''
    if isinstance(ctx, mx.Context):
        ctx = [ctx]
    for epoch in tqdm(range(num_epochs)):
        train_l_sum, n = 0.0, 0
        # train
        for i,batch in enumerate(train_iter):
            Xs, ys, batch_size = _get_batch(batch, ctx)
            ls = []
            with autograd.record():
                y_hats = [net(X) for X in Xs]
                # softmax-cross-entropy loss
                ls = [loss(y_hat,y) for y_hat,y in zip(y_hats,ys)][0]
            ls.backward()
            trainer.step(batch_size)
            train_l_sum += ls.sum().asscalar()
            n += batch_size
        
        # evaluate
        for i,batch in enumerate(iter_test):
            Xs,ys,batch_size = _get_batch(batch, d2l.try_all_gpus())
            if i == 0:
                y_pred = [net(X) for X in Xs][0]
                y_true = ys[0]
            else:
                y_pred = nd.concat(y_pred, [net(X) for X in Xs][0], dim=0)
                y_true = nd.concat(y_true, ys[0], dim=0)
        fpr, tpr, thr = roc_curve(y_true.asnumpy(), softmax(y_pred).asnumpy())
        best_index = get_max_index(tpr-fpr,1)
        best_fpr, best_tpr, best_thr = fpr[best_index], tpr[best_index], thr[best_index]
        roc_auc = auc(fpr, tpr)
        acc = get_acc(softmax(y_pred).asnumpy(), y_true.asnumpy(), best_thr)
        print(epoch+1,best_fpr,best_tpr,best_thr,roc_auc,acc)