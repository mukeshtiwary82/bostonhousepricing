#!/usr/bin/env python
# coding: utf-8

# In[4]:


import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
get_ipython().run_line_magic('matplotlib', 'inline')


# ## lets load the bosten house pricing dataset

# In[5]:


from sklearn.datasets import load_boston


# In[17]:


boston=load_boston()


# In[18]:


type(boston)


# In[19]:


boston.keys()


# ## lets check description of the datasets 

# In[20]:


print(boston.DESCR)


# In[21]:


print(boston.data)


# In[22]:


print(boston.target)


# In[24]:


print(boston.feature_names)


# ## prepearing the dataset

# In[29]:


dataset=pd.DataFrame(boston.data)


# In[30]:


dataset


# In[35]:


dataset=pd.DataFrame(boston.data,columns=boston.feature_names)


# In[36]:


dataset


# In[37]:


dataset.head()


# In[38]:


dataset['price']=boston.target


# In[40]:


dataset.head()


# In[41]:


dataset.info()


# ## summarizing the stats of the data

# In[42]:


dataset.describe()


# ## check missing value

# In[43]:


dataset.isnull()


# In[44]:


dataset.isnull().sum()


# ## exploratory data analysis(EDA)
# ## CORRELATION

# In[45]:


dataset.corr()


# In[46]:


import seaborn as sns
sns.pairplot(dataset)


# In[50]:


plt.scatter(dataset['CRIM'],dataset['price'])
plt.xlabel("Crime Rate")
plt.ylabel("Price")


# In[51]:


plt.scatter(dataset['RM'],dataset['price'])
plt.xlabel("RM")
plt.ylabel("Price")


# In[53]:


import seaborn as sns
sns.regplot(x="RM",y="price",data=dataset)


# In[55]:


sns.regplot(x="LSTAT",y="price",data=dataset)


# In[56]:


sns.regplot(x="CHAS",y="price",data=dataset)


# In[57]:


sns.regplot(x="PTRATIO",y="price",data=dataset)


# In[60]:


## Independent and Dependent features
x=dataset.iloc[:,:-1]
y=dataset.iloc[:,-1]


# In[61]:


x.head()


# In[63]:


y


# In[72]:


## Train Test split
from sklearn.model_selection import train_test_split
x_train,x_test,y_train,y_test=train_test_split(x,y,test_size=0.3,random_state=42)


# In[74]:


x_train


# In[79]:


## standarize the dataset
from sklearn.preprocessing import StandardScaler
scaler=StandardScaler()


# In[82]:


x_train=scaler.fit_transform(x_train)


# In[86]:


x_test=scaler.transform(x_test)


# In[87]:


import pickle
pickle.dump(scaler,open('scaling.pkl','wb'))


# In[88]:


x_train


# In[89]:


x_test


# ## Model Traing

# In[90]:


from sklearn.linear_model import LinearRegression


# In[92]:


regression=LinearRegression()


# In[93]:


regression.fit(x_train,y_train)


# In[94]:


## print the coefficients and the intercept
print(regression.coef_)


# In[95]:


print(regression.intercept_)


# In[96]:


## on which parameter the model has been trained
regression.get_params()


# In[97]:


## prediction with test data
reg_pred=regression.predict(x_test)


# In[98]:


reg_pred


# In[ ]:


## Assumptions
## plot a scatter plot for the prediction
plt.scatter(y_test,reg_pred)


# In[111]:


## Residuals
residuals=y_test-reg_pred


# In[112]:


residuals


# In[113]:


## plot the residuals
sns.displot(residuals,kind="kde")


# In[114]:


## scatter plot with respect to prediction and residuals
##  Uniform distribution
plt.scatter(reg_pred,residuals)


# In[115]:


from sklearn.metrics import mean_squared_error
from sklearn.metrics import mean_absolute_error

print(mean_absolute_error(y_test,reg_pred))
print(mean_squared_error(y_test,reg_pred))
print(np.sqrt(mean_squared_error(y_test,reg_pred)))


# ##  R squared  and adjusted R square
# ## formula
# ## R^2=1-SSR/SST
# ## r^2= coefficient of determination SSR = sum of squqres of residuals SST=total sum of squres

# In[117]:


from sklearn.metrics import r2_score
score=r2_score(y_test,reg_pred)
print(score)


# ## Adjusted R2=1-[(1-R2)*(n-1)/(n-k-1)]
# ## where:
# ## R2: The R2 of the model n:The number of observations k: The number of predictor variables

# In[118]:


## display adjusted R- squared
1-(1-score)*(len(y_test)-1)/(len(y_test)-x_test.shape[1]-1)


# ## NEW DATA PREDICTION

# In[120]:


boston.data[0].reshape(1,-1)


# In[124]:


## transformation of new data
scaler.transform(boston.data[0].reshape(1,-1))


# In[127]:


regression.predict(scaler.transform(boston.data[0].reshape(1,-1)))


# # Picking the model file for deployment

# In[128]:


import pickle


# In[129]:


pickle.dump(regression,open('regmodel.pkl','wb'))


# In[132]:


pickled_model=pickle.load(open('regmodel.pkl','rb'))


# In[133]:


pickled_model.predict(scaler.transform(boston.data[0].reshape(1,-1)))


# In[ ]:




