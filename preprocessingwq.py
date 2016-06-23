#! /usr/bin/env/ python

import csv
import itertools
import operator
import numpy as np
import nltk
import os
import time
import timeit
from datetime import datetime
from utils import *
from rnn_theano import RNNTheano
import sys
vocabulary_size = 8000
unknown_token = 'UNKNOWN_TOKEN'
sentence_start_token ='SENTENCE_START'
sentence_end_token ='SENTENCE_END'

#read data and append SENTENCE_START and SENTENCE_END tokens
print "Reading CSV files..."

with open('data/reddit-comments-2015-08.csv','rb') as f:
    reader = csv.reader(f,skipinitialspace=True)
    reader.next()
    #split full commends into sentences
    #for x in reader:
     #   print [nltk.sent_tokenize(x[0].decode('utf-8').lower())]
     #   break
    sentences = itertools.chain(*[nltk.sent_tokenize(x[0].decode('utf-8').lower()) for x in reader])
    #Append SENTENCE_START and SENTENCE_END
    sentences = ["%s %s %s" % (sentence_start_token,x,sentence_end_token) for x in sentences]
print "Parsed %d sentences." % (len(sentences))

#tokenize the sentences into words
tokenized_sentences = [nltk.word_tokenize(sent) for sent in sentences]

#Count the word frequencies
word_freq = nltk.FreqDist(itertools.chain(*tokenized_sentences))
print "Found %d unique words tokens." % len(word_freq.items())

#Get the most common words and build index_to_word and word_to_index vectors
vocab = word_freq.most_common(vocabulary_size-1)
index_to_word = [x[0] for x in vocab]
index_to_word.append(unknown_token)
print index_to_word
word_to_index = dict([(w,i) for i,w in enumerate(index_to_word)])
print word_to_index
print "Using vocabulary size %d." % vocabulary_size
print  "The least frequent word in our vocabulary is '%s' and appeared %d times." % (vocab[-1][0],vocab[-1][1])

#Replace all words not in our vocabulary with the unknown token
for i,sent in enumerate(tokenized_sentences):
    tokenized_sentences[i] = [w if w in word_to_index else unknown_token for w in sent ]

print "\nExample sentence: '%s'" % sentences[0]
print "\nExample sentece after Pre-processing:'%s'" % tokenized_sentences[0]

#Create the training data

X_train = np.asarray([[word_to_index[w] for w in sent[:-1]] for sent in tokenized_sentences])
Y_train = np.asarray([[word_to_index[w] for w in sent[1:]]for sent in tokenized_sentences])

print X_train[0]
print Y_train[0]

#build the Recurrent Neural Network

#use one-hot vector represents each word.
#Initialization
#first we start by declaring a RNN class an initializing our parameters.
class RNNNumpy:
    def __init__(self,word_dim,hidden_dim=100,bptt_truncate=4):
        #Assian instance variables
        self.word_dim = word_dim
        self.hidden_dim = hidden_dim
        self.bptt_truncate = bptt_truncate
        #randomly initialize the network parameters
        self.U = np.random.uniform(-np.sqrt(1./word_dim),np.sqrt(1./word_dim),(hidden_dim,word_dim))
        self.V = np.random.uniform(-np.sqrt(1./hidden_dim),np.sqrt(1./hidden_dim),(word_dim,hidden_dim))
        self.W = np.random.uniform(-np.sqrt(1./hidden_dim),np.sqrt(1./hidden_dim),(hidden_dim,hidden_dim))

    def forward_propagation(self,x):
        #The total number of time steps
        T = len(x)
        #During forward propagation we save all hidden states in s because need them later
        #We add one additional timestep for the initial hidden, which we set to 0
        s = np.zeros((T+1,self.hidden_dim))
        s[-1] = np.zeros(self.hidden_dim)
        #The outputs at each time step. Again, we save them for later
        o = np.zeros((T,self.word_dim))
        #For each timestep...
        for t in np.arange(T):
            #Note that we are indexing U by x[t].This is the same as multiplying
            #U with a one-hot vector
            s[t] = np.tanh(self.U[:,x[t]]+self.W.dot(s[t-1]))
            o[t] = softmax(self.V.dot(s[t]))
        return [o,s]
    #RNNNumpy.forward_propagation = forward_propagation

    #each o_t is a vector of probabilities representing the words in our vocabulary
    #but sometimes, for example when evaluating our model, all we
    #want is the next word with highest probability. We call this function
    # predict:
    def predict(self,x):
        #Perform forward propagation and return index of the highest score
        o,s = self.forward_propagation(x)
        return np.argmax(o,axis=1)

    #caculating the loss
    def calculate_total_loss(self,x,y):
        L = 0
        #for each sentence, there are T time steps.
        #len(y[i]) == T
        for i in np.arange(len(y)):
            #compute the forward propagation.
            o,s = self.forward_propagation(x[i])
            #o is T*vovabulary_size. We use y[i] as the index of o
            correct_word_predictions = o[np.arange(len(y[i])),y[i]]
            #Add to the loss based on how off we were
            L += -1*np.sum(np.log(correct_word_predictions))
        return L

    def calculate_loss(self,x,y):
        #Divide the total loss by the number of training examples.
        N = np.sum(len(y_i) for y_i in y)
        return self.calculate_total_loss(x,y)/N

    #training the RNN with Stochastic Gradient descend and Backpropagation through time(BPTT)
    def bptt(self,x,y):
        T = len(y)
        #Perform forward propagation
        o,s = self.forward_propagation(x)
        #We accumulate the gradients in these variables.
        dLdU = np.zeros(self.U.shape)
        dLdV = np.zeros(self.V.shape)
        dLdW = np.zeros(self.W.shape)
        #compute the gradient over L with respect to output o
        #We use y to indexing the elements of output o
        delta_o = o
        #delta_o is equal to dLd
        delta_o[np.arange(len(y)),y] -=1.
        #For each output backwards
        for t in np.arange(T)[::-1]:
            dLdV += np.outer(delta_o[t],s[t].T)
            #initial delta calculation
            delta_t = self.V.T.dot(delta_o[t])*(1-(s[t]**2))
            #Backpropagation through time (for at most self.bptt_truncate steps)
            for bptt_step in np.arange(max(0,t-self.bptt_truncate),t+1)[::-1]:
                #print "Backpropagation step t=%d bptt step=%d " % (t,bptt_step)
                dLdW += np.outer(delta_t,s[bptt_step-1])
                dLdU[:,x[bptt_step]] += delta_t
                #Update delta for next step
                delta_t = self.W.T.dot(delta_t)*(1-s[bptt_step]**2)
        return [dLdU,dLdV,dLdW]

    #Gradient checking
    def gradient_check(self,x,y,h = 0.00001,error_threshold=0.01):
        #Calculate the gradients using backpropagation. We want to check if these are correct.
        bptt_gradients = self.bptt(x,y)
        #List of all parameters we want to check.
        model_parameters = ['U','V','W']
        #Gradient check for each parameter
        for pidx,pname in enumerate(model_parameters):
            #Get the actual parameter value from the model, e.g. model.W
            parameter = operator.attrgetter(pname)(self)
            print "Performing gradient check for parameter %s with size %d" % (pname,np.prod(parameter.shape))
            #Iterate over each element of the parameter matrix, e.g. (0,0),(0,1),
            it = np.nditer(parameter,flags=['multi_index'],op_flags=['readwrite'])
            while not it.finished:
                ix = it.multi_index
                #save the original value so we can reset it later
                original_value = parameter[ix]
                #Estimate the gradient using (f(x+h)-f(x-h))/2*h
                parameter[ix] = original_value +h
                gradplus = self.calculate_total_loss([x],[y])
                parameter[ix] = original_value-h
                gradminus = self.calculate_total_loss([x],[y])
                estimated_gradient = (gradplus-gradminus)/(2*h)
                #Reset parameter to original value
                parameter[ix] = original_value
                #The gradient for this parameter calculated using backpropagation
                backprop_gradient = bptt_gradients[pidx][ix]
                #calculate the relative error
                relative_error = np.abs(backprop_gradient-estimated_gradient)/(np.abs(backprop_gradient)+np.abs(estimated_gradient))
                #if the error is too large fail the gradient check
                if relative_error > error_threshold:
                    print "Gradient check ERROR: parameter=%s ix=%s " %(pname,ix)
                    print "+h Loss: %f " % gradplus
                    print "-h Loss:%f " % gradminus
                    print "Estimated_gradient: %f " % estimated_gradient
                    print "Backpropagation gradient:%f " % backprop_gradient
                    print "Relative Error:%f " % relative_error
                    return
                it.iternext()
            print "Gradient check for parameter %s passed." % (pname)

    #SGD Implementation
    #Now that we are able to calculate the gradients for our parameters we can implement SGD.
    #I like to do this in two steps:1.A function sdg_step that calculates the gradients
    #and performs the updates for one batch.
    #2.An outer lood that iterates through the training set and adjusts the learning rate.
    #Performs one step of SGD
    def numpy_sgd_step(self,x,y,learning_rate):
        #Calculate the gradients
        dLdU,dLdV,dLdW = self.bptt(x,y)
        #Change paramters according to gradients and learning rate
        self.U = self.U - learning_rate*dLdU
        self.V = self.V -learning_rate*dLdV
        self.W = self.W - learning_rate*dLdW
    #define the outer loop method
    # - model: The RNN model instance
    # - X_train: the training data set
    # - Y_train: the training labels
    # - learning_rate: Initial learning rate for SGD
    # - nepoch:Number of times to iterate through the complete dataset
    # - evaluate_loss_after: Evaluate the loss after this many epochs
def traing_with_sgd(model,X_train,Y_train,learning_rate=0.005,nepoch = 100,evaluate_loss_after = 5):
    losses = []
    num_example_seen = 0
    for epoch in range(nepoch):
        #Optionally evaluate the loss
        if (epoch % evaluate_loss_after == 0):
            loss = model.calculate_loss(X_train,Y_train)
            losses.append((num_example_seen,loss))
            time = datetime.now().strftime('%Y-%m-%d %H:%M:%S')
            print "%s: Loss after num_examples_seen=%d, epoch = %d: %f " % (time,num_example_seen,epoch,loss)
            #Adjust the learning rate if loss increases
            if (len(losses)>1 and losses[-1][1] > losses[-2][1]):
                learning_rate = learning_rate*0.5
                print "Setting learning rate to %f" % learning_rate
            sys.stdout.flush()
        #for each training example
        for i in range(len(Y_train)):
            #One SGD step
            model.numpy_sgd_step(X_train[i],Y_train[i],learning_rate)
            num_example_seen +=1
    return losses
























