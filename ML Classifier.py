#!/usr/bin/env python
# coding: utf-8

# In[1]:


import sklearn


# In[2]:


from sklearn.datasets import load_breast_cancer


# In[3]:


data=load_breast_cancer()


# In[4]:


label_names=data['target names']
labels=data['target']
feature_names=data['feature_names']
features=data['data']


# In[5]:


label_names=data['target_names']
labels=data['target']
feature_names=data['feature_names']
features=data['data']


# In[6]:


print(label_names)
print(labels[0])
print(feature_names[0])
print(features[0])


# In[7]:


from sklearn.model_selection import train_test_split


# In[8]:


#split the data
train, test, train_labels, test_labels = train_test_split(features, labels, test_size=0.33,random_state=42)


# In[9]:


from sklearn.naive_bayes import GaussianNB


# In[10]:


gnb= GaussianNB()


# In[11]:


model = gnb.fit(train, train_labels)


# In[12]:


#make predictions
preds = gnb.predict(test)
print(preds)


# In[13]:


from sklearn.metrics import accuracy_score
#evaluate accuracy
print(accuracy_score(test_labels, preds))


# In[ ]:




