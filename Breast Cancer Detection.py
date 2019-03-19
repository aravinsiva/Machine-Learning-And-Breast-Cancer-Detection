#!/usr/bin/env python
# coding: utf-8

# In[1]:


import sys
import numpy
import matplotlib
import pandas
import sklearn

print('Python {}'.format(sys.version))


# In[ ]:





# In[34]:


import numpy as np
from sklearn import preprocessing
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
from sklearn import model_selection
from sklearn.metrics import classification_report, accuracy_score
from pandas.plotting import scatter_matrix
import matplotlib.pyplot as plt
import pandas as pd
 


# In[35]:


#Loading dataset
url = "https://archive.ics.uci.edu/ml/machine-learning-databases/breast-cancer-wisconsin/breast-cancer-wisconsin.data"
names= ['id','clump_thickness','univorm_cell_size','uniform_cell_shape', 'marginal_adhesion', 'single_epethelial_size',
        'bare_nuclei','bland_chromatin','normal_nucleoli','mitosis','class']

df=pd.read_csv(url,names=names)


# In[36]:


df.replace('?',-99999,inplace=True)
print(df.axes)

df.drop(['id'],1,inplace=True)

#Print Shae of Data
print(df.shape)


# In[38]:


#Dataset Visualization
print (df.loc[698])
print(df.describe())


# In[39]:


#Plot histogram

df.hist(figsize=(10,10))
plt.show()


# In[40]:


#Create a scatter plot matrix
scatter_matrix(df,figsize=(18,18))
plt.show()


# In[56]:


#Create X & Y datasets for training

X=np.array(df.drop(['class'],1))
y=np.array(df['class'])

X_train,X_test,y_train,y_test=model_selection.train_test_split(X,y,test_size=0.2)


# In[57]:


#Specify options

seed=8
scoring='accuracy'


# In[58]:


models=[]
models.append(('KNN',KNeighborsClassifier(n_neighbors=5)))
models.append(('SVM',SVC()))

#Evaluate model

results=[]
names=[]


for name, model in models:
    kfold = model_selection.KFold(n_splits=10,random_state=seed)
    cv_results= model_selection.cross_val_score(model,X_train,y_train,cv=kfold,scoring=scoring)
    
    results.append(cv_results)
    names.append(name)
    print (name)
    print(cv_results.mean())
    print(cv_results.std())


# In[59]:


#Make predictions on validation data set

for name, model in models:
    model.fit(X_train,y_train)
    predictions=model.predict(X_test)
    print(name)
    print(accuracy_score(y_test,predictions))
    print (classification_report(y_test,predictions))


# In[60]:


clf=SVC()

clf.fit(X_train,y_train)
accuracy=clf.score(X_test,y_test)
print(accuracy)

example=np.array([[4,2,1,1,1,2,3,2,10]])
example= example.reshape(len(example),-1)
prediction= clf.predict(example)
print(prediction)


# In[61]:



if (prediction==4):
    print ("Tumour is Malignant!!!!")
else:
    print("Tumour is Benign")


# In[ ]:




