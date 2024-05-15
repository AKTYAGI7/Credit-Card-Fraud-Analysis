#!/usr/bin/env python
# coding: utf-8

# # Credit Card Fraud Analysis

# In[1]:


import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score


# In[ ]:





# In[4]:


credit_card_data = pd.read_csv('creditcard.csv')


# In[5]:


credit_card_data.head()


# In[6]:


credit_card_data.tail()


# In[7]:


credit_card_data.info()


# In[8]:


credit_card_data.isnull().sum()


# In[9]:


credit_card_data['Class'].value_counts()


# In[10]:


legit = credit_card_data[credit_card_data.Class == 0]
fraud = credit_card_data[credit_card_data.Class == 1]


# In[11]:


print(legit.shape)
print(fraud.shape)


# In[12]:


legit.Amount.describe()


# In[13]:


fraud.Amount.describe()


# In[14]:


credit_card_data.groupby('Class').mean()


# In[15]:


legit_sample = legit.sample(n=492)


# In[16]:


new_dataset = pd.concat([legit_sample, fraud], axis=0)


# In[17]:


new_dataset.head()


# In[18]:


new_dataset.tail()


# In[19]:


new_dataset['Class'].value_counts()


# In[20]:


new_dataset.groupby('Class').mean()


# In[21]:


X = new_dataset.drop(columns='Class', axis=1)
Y = new_dataset['Class']


# In[22]:


print(X)


# In[24]:


print(Y)


# In[25]:


X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.2, stratify=Y, random_state=2)


# In[26]:


print(X.shape, X_train.shape, X_test.shape)


# In[27]:


model = LogisticRegression()


# In[28]:


model.fit(X_train, Y_train)


# In[29]:


X_train_prediction = model.predict(X_train)
training_data_accuracy = accuracy_score(X_train_prediction, Y_train)


# In[30]:


print('Accuracy on Training data : ', training_data_accuracy)


# In[31]:


X_test_prediction = model.predict(X_test)
test_data_accuracy = accuracy_score(X_test_prediction, Y_test)


# In[32]:


print('Accuracy score on Test Data : ', test_data_accuracy)


# In[ ]:




