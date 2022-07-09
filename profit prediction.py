#!/usr/bin/env python
# coding: utf-8

# In[1]:


import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt


# In[2]:


data = pd.read_excel("C:/Users/gayat/Downloads/50_Startups (1).xlsx")


# In[3]:


data


# In[4]:


data.isnull().sum()


# In[5]:


data.describe()


# In[6]:


data.dtypes


# In[30]:


#encoding
from sklearn import preprocessing
label_encoder = preprocessing.LabelEncoder()
data['State'] = label_encoder.fit_transform(data['State'])


# In[31]:


X = data.iloc[: ,: -1]


# In[32]:


X


# In[33]:


y = data.iloc[:, -1 :]


# In[34]:


y


# In[35]:


#finding outliers
def remove_outlier_IQR(data):
    Q1 = data.quantile(0.25)
    Q3 = data.quantile(0.75)
    IQR = Q1-Q3
    data_final = data[~(data<(Q1-1.5*IQR)) | (data>(Q3+1.5*IQR))]
    return data_final


# In[36]:


#Training
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X,y, test_size=0.5, random_state=0)


# In[37]:


#normalization
from sklearn.preprocessing import StandardScaler
sc = StandardScaler()
X_train = sc.fit_transform(X_train)
X_test = sc.fit_transform(X_test)


# In[38]:


#performing model
from sklearn.linear_model import LinearRegression
algo = LinearRegression()
algo.fit(X_train, y_train)


# In[39]:


y_predict = algo.predict(X_test)


# In[40]:


y_test


# In[41]:


y_predict


# In[ ]:




