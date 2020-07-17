#!/usr/bin/env python
# coding: utf-8

# In[1]:


import numpy as np 
import pandas as pd 
import re
import nltk
from matplotlib import pyplot as plt
from nltk.corpus import stopwords
from nltk.stem.porter import PorterStemmer
from sklearn.metrics import f1_score
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
import matplotlib.pyplot as plt 
from wordcloud import WordCloud, STOPWORDS 


# In[ ]:



#https://www.kaggle.com/c/nlp-getting-started/overview


# In[2]:


#Reading training and testing datasets 
ds_train=pd.read_csv("train.csv")
ds_test=pd.read_csv("test.csv")


# In[3]:


#printing training dataset
ds_train.head()
print(len(ds_train))


# In[4]:


#Calculating NaN values.
train_nans= ds_train['keyword'].isnull().sum()
print(train_nans)
train_nans = ds_train['location'].isnull().sum()
print(train_nans)
test_nans= ds_test['keyword'].isnull().sum()
print(test_nans)
test_nans = ds_test['location'].isnull().sum()
print(test_nans)


# In[5]:


# Replacing nans.
ds_train['keyword'].fillna('none', inplace=True)
ds_train['location'].fillna('Unknown', inplace=True)
ds_test['keyword'].fillna('none', inplace=True)
ds_test['location'].fillna('Unknown', inplace=True)


# In[6]:


# Checking NaN values again.
ds_train.head()
train_nans= ds_train['keyword'].isnull().sum()
print(train_nans)
train_nans = ds_train['location'].isnull().sum()
print(train_nans)


# In[7]:


print(ds_train.shape[0])


# In[ ]:





# In[8]:


# Creating Corpus after preprocessing the training data
corpus  = []
pstem = PorterStemmer()
for i in range(ds_train['text'].shape[0]):
    text = re.sub("[^a-zA-Z]", ' ', ds_train['text'][i])
    text = text.lower()
    text = text.split()
    text = [pstem.stem(word) for word in text if not word in set(stopwords.words('english'))]
    text = ' '.join(text)
    corpus.append(text)
    


# In[9]:


#print((corpus))


# In[10]:


corpus


# In[11]:


#Create dictionary based on corpus 
uniqueWords = {}
for text in corpus:
    for word in text.split():
        
        if(word in uniqueWords.keys()):
            uniqueWords[word] = uniqueWords[word] + 1
        else:
            uniqueWords[word] = 1


# In[12]:


print(uniqueWords)


# In[ ]:





# In[ ]:





# In[13]:


print(len(uniqueWords))


# In[14]:


#Converting dictionary to dataFrame
uniqueWords = pd.DataFrame.from_dict(uniqueWords,orient='index',columns=['WordFrequency'])
uniqueWords.sort_values(by=['WordFrequency'], inplace=True, ascending=False)
#print(uniqueWords)
print("Number of records in Unique Words Data frame are ")
print((len(uniqueWords)))
uniqueWords.head(10)


# In[17]:


# Creating the Bag of Words model by vectorizing the input data
from sklearn.feature_extraction.text import CountVectorizer
cv = CountVectorizer(max_features = len(uniqueWords))
X = cv.fit_transform(corpus).todense()
y = ds_train['target'].values


# In[ ]:





# In[18]:


print(X.shape[0])


# In[19]:


#Split the training data set to train and test data
X_train , X_test , y_train , y_test = train_test_split(X,y,test_size=0.2)
print('Train Data splitted successfully')
print(X_test)


# In[20]:


from sklearn.neighbors import KNeighborsClassifier


# In[22]:


# Applying Model KNN
classifier_knn = KNeighborsClassifier(n_neighbors = 6,weights = 'distance',algorithm = 'brute')
classifier_knn.fit(X_train, y_train)
y_pred_knn = classifier_knn.predict(X_test)


# In[30]:


#Calculating Evaluation Measures
print("K-Nearest Neighbour Model Accuracy Score for Train Data set is")
print((classifier_knn.score(X_train, y_train)))
print("K-Nearest Neighbour Model Accuracy Score for Test Data set is ")
print(classifier_knn.score(X_test, y_test) )   
print("K-Nearest Neighbour Model F1 Score is ")
print((f1_score(y_test, y_pred_knn)))


# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:




