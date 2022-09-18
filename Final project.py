#!/usr/bin/env python
# coding: utf-8

# In[3]:


import numpy as np 
import pandas as pd 
import seaborn as sns                 
import matplotlib.pyplot as plt       
get_ipython().run_line_magic('matplotlib', 'inline')
import warnings                        
warnings.filterwarnings("ignore")


# In[4]:


import os
for dirname, _, filenames in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))


# In[5]:


train = pd.read_csv('train.csv')
test = pd.read_csv('test.csv')


# In[6]:


train.columns


# In[7]:


test.columns


# In[8]:


print(train.shape) 
print(test.shape)


# In[9]:


print(train.dtypes)


# In[10]:


train['subscribed'].value_counts()


# In[11]:


train['subscribed'].value_counts(normalize=True)


# In[12]:


train['subscribed'].value_counts().plot.bar(color=['red','green'])


# In[13]:


sns.distplot(train["age"],color='green')


# In[14]:


train['job'].value_counts().plot.bar(color=['red', 'hotpink', 'green', 'blue', 'cyan','yellow','magenta'])


# In[15]:


train['default'].value_counts().plot.bar(color = ['blue','red'])


# In[16]:


print(pd.crosstab(train['job'],train['subscribed']))

job=pd.crosstab(train['job'],train['subscribed'])
job.div(job.sum(1).astype(float), axis=0).plot(kind="bar", stacked=True, figsize=(8,8))
plt.xlabel('Job')
plt.ylabel('Percentage')


# In[17]:


print(pd.crosstab(train['default'],train['subscribed']))

default=pd.crosstab(train['default'],train['subscribed'])
default.div(default.sum(1).astype(float), axis=0).plot(kind="bar", stacked=True, figsize=(8,8))
plt.xlabel('default')
plt.ylabel('Percentage')


# In[18]:


train['subscribed'].replace('no', 0,inplace=True)
train['subscribed'].replace('yes', 1,inplace=True)


# In[19]:


corr = train.corr()
mask = np.array(corr)
mask[np.tril_indices_from(mask)] = False
fig,ax= plt.subplots()
fig.set_size_inches(20,10)
sns.heatmap(corr, mask=mask,vmax=.9, square=True,annot=True, cmap="YlGnBu")


# In[20]:


train.isnull().sum() 


# In[21]:


target = train['subscribed']
train = train.drop('subscribed',1)


# In[22]:


train = pd.get_dummies(train)


# In[23]:



from sklearn.model_selection import train_test_split


# In[24]:


X_train, X_val, y_train, y_val = train_test_split(train, target, test_size = 0.2, random_state=12)


# In[ ]:





# In[25]:


from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score


# In[26]:


model = LogisticRegression()


# In[27]:


model.fit(X_train,y_train)


# In[28]:


prediction = model.predict(X_val)


# In[29]:


accuracy_score(y_val, prediction)


# In[ ]:





# In[30]:


from sklearn.tree import DecisionTreeClassifier


# In[31]:


clf = DecisionTreeClassifier(max_depth=4, random_state=0)


# In[32]:


clf.fit(X_train,y_train)


# In[33]:


predict = clf.predict(X_val)


# In[34]:


accuracy_score(y_val, predict)


# In[35]:


test = pd.get_dummies(test)


# In[36]:


test_prediction = clf.predict(test)


# In[37]:


submission = pd.DataFrame()


# In[38]:


submission['ID'] = test['ID']
submission['subscribed'] = test_prediction


# In[39]:


submission['subscribed'].replace(0,'no',inplace=True)
submission['subscribed'].replace(1,'yes',inplace=True)


# In[40]:


submission.to_csv('submission.csv', header=True, index=False)


# In[41]:


pip install plotting


# In[42]:


import seaborn as sns
import matplotlib.pyplot as plt


# In[44]:


get_ipython().system('pip install --upgrade pandas')


# In[45]:


pd.plot(kind="bar", x="age", y="marital")


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





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:




