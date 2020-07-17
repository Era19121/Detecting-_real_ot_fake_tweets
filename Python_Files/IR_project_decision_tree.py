# -*- coding: utf-8 -*-
"""IR_project_Decision_Tree.ipynb

Automatically generated by Colaboratory.

Original file is located at
    https://colab.research.google.com/drive/1MmtxJAoKG1P_VHks0hQBnT8SjSe7DRat
"""

import numpy as np
import pandas as pd
import nltk
import re
import string
from nltk.stem import PorterStemmer
from nltk.corpus import stopwords 
from collections import defaultdict
from num2words import num2words
from nltk.tokenize import word_tokenize 
import collections
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
import seaborn
from sklearn.metrics import f1_score
from sklearn.metrics import accuracy_score
import math
import json
from sklearn.metrics import confusion_matrix
from sklearn.metrics import accuracy_score
from sklearn.decomposition import TruncatedSVD

#converting traning.csv to dataframe
datatrain=pd.read_csv(r"C:\Users\Ashwani Sharma\Desktop\Era\IIIT DELHI\SEM2\IR\project\train.csv")

datatrain

#filling NaN with none for keyword  and unknown for location
datatrain.location.fillna("unknown",inplace=True)
datatrain.keyword.fillna("none",inplace=True)
datatrain

#preprocessing training documents and creating [document][term] vocabulary named "vocab"
idSent=datatrain[['id','text']]
vocab=defaultdict(list)
num=0
tokens=defaultdict(list)
freq=0
sentence=[]
new_train=pd.DataFrame(columns=["id",'processed_text'])
portstem=PorterStemmer()
tweets=[]
for ID in idSent['id']:
    vocab[ID]={}
    words=[]
         
    while(num!=len(datatrain)):
        stopw = set(stopwords.words('english'))
           
        vocabwords=nltk.tokenize.word_tokenize((idSent['text'][num]).lower())
        
        #print(vocabwords)
        for w in vocabwords:
            w=w.strip("=")
            w=w.strip("'")
            w=w.strip('"')
            w=w.strip(".")
            w=w.strip(":")
            w=w.strip(";")
            w=w.strip("*")
            w=w.strip('')
           
            w=portstem.stem(w)
            if(w.isdigit()):
                w=num2words(w)
            
            
            if w not in string.punctuation and w not in stopw:
                
                if w not in tokens and w not in vocab[ID] :
                    
                    vocab[ID][w]=1
                    tokens[w]=1
                    #print("1")
                    #print(vocab[ID])
                elif w in tokens:
                    tokens[w]=tokens[w]+1
                    if w not in vocab[ID]:
                        #print("2")
                        vocab[ID][w]=1
                        
                        #print(vocab[ID])    
                    elif w in vocab[ID]:
                        #print("3")
                        vocab[ID][w]=1+vocab[ID][w]
                    #print(vocab[ID])
            #print("4")
            #print(vocab[ID])
            
                words.append(w)
                
        new_train=new_train.append({'id':ID,'processed_text':words},ignore_index=True) 
        sentence.append(words)
            
           
              
        #print(words)    
        
        break
    
    num=num+1
    
for i in sentence:
    tweet=''
    for word in i:
        tweet=tweet+word
        tweet=tweet+' '
        
    tweets.append(tweet) 
#print(new_train)
#print(tweets)
print(vocab)

#frequency of unique words in training corpus
print(tokens)

#visual reprsentation of distribution of target values
seaborn.countplot(data =datatrain, x = "target")

#creating tfidf [term][document] inverted index
def tfidf_vocab2(token,idsent,vocab1):
    vocab2=defaultdict(dict)
    for term in token:

        c=0
        for ID in idsent['id']:

            if term in vocab1[ID]:
                c=c+1
        df=c
    
   
        for ID in idsent['id']:
            #print(term)
            #print(ID)

            vocab2[term][ID]=0
            if term in vocab1[ID]:
                tf=vocab1[ID][term]
                tf=math.log((1+tf),10)
                idf=math.log((len(idsent)/df),10)
                tfidf=tf*idf
                vocab2[term][ID]=tfidf

    #print(vocab2)
    #with open("tfidffile.json","w") as file:
    #    file.write(json.dumps(vocab2))
    return vocab2

#calling vocab2 function for training data
vocab2=tfidf_vocab2(tokens,idSent,vocab)

#doc term tfidf matrix
def matrix(token,vocab1,Vocab2):
    
    docmatrix=np.zeros(shape=(len(vocab1),len(token)))


    print(len(vocab1))
    print(len(token))
    i=0
    j=0
    for doc_i in vocab1:
        j=0
        for term_j in token:


            if term_j in vocab1[doc_i]:

                docmatrix[i][j]=Vocab2[term_j][doc_i]

            else:
                docmatrix[i][j]=0



            j=j+1
            if(j==len(token)):
                break
        i=i+1
        if(i==len(vocab1)):
            break

    print(docmatrix)
    return docmatrix

#training decision tree classifier with tfidf matrix
X=matrix(tokens,vocab,vocab2)
Y=datatrain['target']
print("X shape = ",X.shape)
print("y shape = ",Y.shape)

X_tr,X_te,Y_tr,Y_te=train_test_split(X,Y,test_size=0.20, random_state=55, shuffle =True)

decisionTree= DecisionTreeClassifier(criterion= 'entropy',
                                           max_depth = None, 
                                           splitter='best', 
                                           random_state=55)

decisionTree.fit(X_tr,Y_tr)
#predicting values
Y_pred=decisionTree.predict(X_te)
print("F1score=",f1_score(Y_te,Y_pred))
print("accuracy=",accuracy_score(Y_te,Y_pred))

#================= test.csv ============================

#converting test.csv to dataframe
datatest=pd.read_csv(r"C:\Users\Ashwani Sharma\Desktop\Era\IIIT DELHI\SEM2\IR\project\test.csv")
datatest

#filling NaN with none for keyword  and unknown for location
datatest.location.fillna("unknown",inplace=True)
datatest.keyword.fillna("none",inplace=True)
datatest

#preprocessing test documents and creating [document][term] vocabulary named "vocab_test"
idSent_test=datatest[['id','text']]
vocab_test=defaultdict(list)
num=0
tokens_test=defaultdict(list)
freq=0
sentence=[]
new_test=pd.DataFrame(columns=["id",'processed_text'])
portstem=PorterStemmer()
tweets_test=[]
for ID in idSent_test['id']:
    vocab_test[ID]={}
    words=[]
         
    while(num!=len(datatest)):
        stopw = set(stopwords.words('english'))
           
        vocabwords=nltk.tokenize.word_tokenize((idSent_test['text'][num]).lower())
        
        #print(vocabwords)
        for w in vocabwords:
            w=w.strip("=")
            w=w.strip("'")
            w=w.strip('"')
            w=w.strip(".")
            w=w.strip(":")
            w=w.strip(";")
            w=w.strip("*")
            w=w.strip('')
           
            w=portstem.stem(w)
            if(w.isdigit()):
                w=num2words(w)
            
            
            if w not in string.punctuation and w not in stopw:
                
                if w not in tokens_test and w not in vocab_test[ID] :
                    
                    vocab_test[ID][w]=1
                    tokens_test[w]=1
                    #print("1")
                    #print(vocab[ID])
                elif w in tokens_test:
                    tokens_test[w]=tokens_test[w]+1
                    if w not in vocab_test[ID]:
                        #print("2")
                        vocab_test[ID][w]=1
                        
                        #print(vocab[ID])    
                    elif w in vocab_test[ID]:
                        #print("3")
                        vocab_test[ID][w]=1+vocab_test[ID][w]
                    #print(vocab[ID])
            #print("4")
            #print(vocab[ID])
            
                words.append(w)
                
        new_test=new_test.append({'id':ID,'processed_text':words},ignore_index=True) 
        sentence.append(words)
            
           
              
        #print(words)    
        
        break
    
    num=num+1
    
for i in sentence:
    tweet_test=''
    for word in i:
        tweet_test=tweet_test+word
        tweet_test=tweet_test+' '
        
    tweets_test.append(tweet_test) 
#print(new_train)
#print(tweets)
print(vocab_test)

#prediction of test.csv decision tree
vocab2_test=tfidf_vocab2(tokens_test,idSent_test,vocab_test)
X_test=matrix(tokens,vocab_test,vocab2_test)
decisionTree.fit(X,Y)
Y_pred_test=decisionTree.predict(X_test)

c1=pd.Series(idSent_test['id'])
pred=pd.Series(Y_pred_test)

test_result=pd.DataFrame(c1)
test_result=pd.concat([test_result,pred], axis = 1)
test_result.columns=["ID","target"]
test_result.to_csv(r'C:\Users\Ashwani Sharma\Desktop\Era\IIIT DELHI\SEM2\IR\project\MT19121_test_result_IR_DT.csv', index = False)
