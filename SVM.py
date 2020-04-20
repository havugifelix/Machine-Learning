#!/usr/bin/env python
# coding: utf-8

# ## Step 1: We import all libraries

# In[ ]:





# In[52]:


import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn import svm
from sklearn.preprocessing import StandardScaler


# ## Step 2: Import oour data and visualize it

# In[53]:


df= pd.read_csv('german.data-numeric.csv')
X= df.drop(columns="C")
Y=df["C"]
X_train, X_test, y_train, y_test = train_test_split(X, Y, test_size=0.2, random_state=1, stratify=Y)

sc_X = StandardScaler()
X_train = sc_X.fit_transform(X_train)
X_test = sc_X.transform(X_test)


# ### Construct our classifier and retun F1 score

# In[141]:


from sklearn.metrics import confusion_matrix
from sklearn.metrics import f1_score
import time
label=['Bad', 'Good']
def predict(c,kernel):
    if kernel=="poly":
        model=svm.SVC(C=c,degree=2,kernel=kernel)
    else:
        model=svm.SVC(C=c,kernel=kernel)
    start=time.time()
    model.fit(X_train,y_train)
    stop= time.time()
    
    y_pred=model.predict(X_test)
    f1=f1_score(y_test,y_pred,pos_label=label[1])
    return [f1,(stop-start)]
       
        


# In[143]:


tim=predict(1,"linear")[0]
tim


# ## Step 4: Evaluate the classifier by F-score

# In[146]:


c=[i for i in np.arange(0.001,5,0.5)]
f1_linear=[predict(i,"linear")[0] for i in c]
train_time_lin=[predict(i,"linear")[1] for i in c]
f1_rbf=[predict(i,"rbf")[0] for i in c]
train_time_rbf=[predict(i,"rbf")[1] for i in c]
f1_poly=[predict(i,"poly") for i in c]
train_time_poly=[predict(i,"poly")[1] for i in c]
# t=f1_linear=predict(1,"linear") f
# t


# In[127]:


from matplotlib import pyplot as plt
import numpy as np

fig=plt.figure(figsize=(15,15))
fig.add_subplot(221)
plt.plot(c, f1_linear)
plt.title("F1 score for Linear classifier")
plt.xlabel('Coefficient C')
plt.ylabel('F1 Score')

fig.add_subplot(222)
plt.plot(c, f1_rbf)
plt.title("F1 score for rbf classifier")
plt.xlabel('Coefficient C')
plt.ylabel('F1 Score')

fig.add_subplot(223)
plt.plot(c, f1_poly)
plt.title("F1 score for polynomial classifier")
plt.xlabel('Coefficient C')
plt.ylabel('F1 Score')


# In[132]:


## Calculating 
m=max(f1_linear)
for i in range(len(f1_linear)):
    if f1_linear[i]==m:
        print(c[i])
        print(max(f1_linear))
for i in range(len(f1_linear)):
    if f1_rbf[i]==max(f1_rbf):
        print(c[i])
        print(max(f1_rbf))
for i in range(len(f1_linear)):
    if f1_poly[i]==max(f1_poly):
        print(c[i])
        print(max(f1_poly))


# In[147]:


fig=plt.figure(figsize=(15,15))
fig.add_subplot(221)
plt.plot(c, train_time_lin)
plt.title("training time for linear classfier")
plt.xlabel('Coefficient C')
plt.ylabel('training time')

fig.add_subplot(222)
plt.plot(c, train_time_rbf)
plt.title("Training time for rbf classifier")
plt.xlabel('Coefficient C')
plt.ylabel('Training time')

fig.add_subplot(223)
plt.plot(c, train_time_poly)
plt.title("training time for polynomial classifier")
plt.xlabel('Coefficient C')
plt.ylabel('training time')


# In[ ]:




