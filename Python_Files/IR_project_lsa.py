# -*- coding: utf-8 -*-
"""IR_project_LSA.ipynb

Automatically generated by Colaboratory.

Original file is located at
    https://colab.research.google.com/drive/1IJsCamHqDLVTmcAP6YbjVJBVGvLcVWR7
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

  
    return vocab2

#calling vocab2 function for training data
vocab2=tfidf_vocab2(tokens,idSent,vocab)

# creating [doc][term] tfidf matrix
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

#================LATENT SEMANTIC ANALYSIS============================#

#stemmed document list
doc_list=[]
for i in new_train["processed_text"]:
    doc_list.append(i)

#seperating [doc term] matrix into target 1 entries as topic 1 and target0 entries as topic 0
topic1=np.zeros(shape=(len(vocab),len(tokens)))
topic0=np.zeros(shape=(len(vocab),len(tokens)))
i=0
j=0
for doc_i in vocab:
    if doc_i==len(vocab):
        break
    
    if datatrain.iloc[int(doc_i)]["target"]==1:
        for j in range(len(tokens)):
       
            topic1[i][j]=X[i][j]
           
            j=j+1
    else:
        for j in range(len(tokens)):
            topic0[i][j]=X[i][j]
            
            j=j+1
    
    i=i+1

# extracting words that will represent 2 topics
model_svd = TruncatedSVD(n_components=2, algorithm='randomized', n_iter=90, random_state=100)
model_svd.fit(topic0)
topic_terms={}

j=0
for i, comp in enumerate(model_svd.components_):
    topic_terms[i]=[]
    terms = zip(tokens, comp)
    sortedterms = sorted(terms, key= lambda x:x[1], reverse=True)[:1000]
    for t in sortedterms:
        topic_terms[i].append(t)


if(j==1):
    topic_disas={}
    topic_disas=topic_terms
    
else:
    topic_ndisas={}

    topic_ndisas=topic_terms

#2 topics extracted
print(topic_ndisas)
print("\n")
print(topic_disas)
print("\n")

#classifying docs on the basis of adition of similarity scores of words matched in train document and topic words
# topic 0=disastrous topic 1=non disastrous
def Classification(vocab,topic_terms):
    classification={}
    cal=defaultdict(dict)
    for doc in vocab:
        for topic in topic_terms:
            cal[doc][topic]=0
            for word in vocab[doc]:
                for term in range(len(topic_terms[topic])):
                    if word==topic_terms[topic][term][0]:
                        cal[doc][topic]=cal[doc][topic]+topic_terms[topic][term][1]

        cal[doc]=collections.OrderedDict(sorted(cal[doc].items(),reverse=True,key=lambda x: x[1]))

        for k,v in cal[doc].items():

            classification[doc]=k
            break
    return classification,cal

# finding accuracy and confusion matrix
def ACCURACY(Y,classification):
   
    actual_classification=[]
    precdicted_classification=[]
    for i in Y:
        actual_classification.append(i)
    for i in classification:
        if classification[i]==1:
            precdicted_classification.append(0)
        else:
            precdicted_classification.append(1)

    conf_matrix=confusion_matrix(actual_classification, precdicted_classification)
    accuracy=accuracy_score(actual_classification, precdicted_classification)
    print(conf_matrix)
    print(accuracy)

# printing topic words

topic_terms={}
topic_terms[0]=topic_ndisas[1]
print("topic_disasasterous",topic_terms[0])

print("\n")
topic_terms[1]=topic_disas[0]
print("topic_non disasastereous",topic_terms[1])
print(len(topic_terms[0]))
print(len(topic_terms[1]))
print("\n"+"predicting ")
classification_train,cal_train=Classification(vocab,topic_terms)
print("\n"+"accuracy")
ACCURACY(Y,classification_train)

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

#prediction of test.csv lsa
classification_test,cal_test=Classification(vocab_test,topic_terms)
precdicted_classification=[]
for i in classification_test:
    if classification_test[i]==1:
        precdicted_classification.append(0)
    else:
        precdicted_classification.append(1)
        
c1=pd.Series(idSent_test['id'])
test_result=pd.DataFrame(c1)
pred=pd.Series(precdicted_classification)
test_result=pd.concat([test_result,pred], axis = 1)
test_result.columns=["ID","target"]
test_result.to_csv(r'C:\Users\Ashwani Sharma\Desktop\Era\IIIT DELHI\SEM2\IR\project\MT19121_test_result_IR_LSA.csv', index = False)

#saving similarity scores
with open("train_dict.json","w") as file:
    file.write(json.dumps(cal))
with open("test_dict.json","w") as file:
    file.write(json.dumps(cal_test))