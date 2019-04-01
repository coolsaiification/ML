#!/usr/bin/env python
# coding: utf-8

# # Non-Linear Classification
# 
# - In many real life problems, the data is not linearly separable,but we need to classify the data. This can be done using by projecting the data to higer dimesions so that it becomes linearly separable.
# <img src="https://github.com/coding-blocks-archives/machine-learning-online-2018/blob/master/12.%20Support%20Vector%20Machines/img/linearly_separable.PNG?raw=true" alt="Linear Separable" style="width: 600px;"/>

# ##  Projecting data to higher dimensions!
# When working with non-linear datasets, we can project orginal feature vectors into higher dimensional space where they can be linearly separated!  
# 
# ### Let us see one example
# 
# 
# Data in 2-Dimensional Space
# <img src="https://github.com/coding-blocks-archives/machine-learning-online-2018/blob/master/12.%20Support%20Vector%20Machines/img/circles_low.png?raw=true" alt="Linear Separable" style="width: 400px;"/>
# 
# Data Projected in 3-D Dimensional Space, after processing the original data using a non-linear function.
# <img src="https://github.com/coding-blocks-archives/machine-learning-online-2018/blob/master/12.%20Support%20Vector%20Machines/img/circles_3d.png?raw=true" alt="Linear Separable" style="width: 400px;"/>

# In[1]:


from sklearn.datasets import make_circles
import numpy as np
import matplotlib.pyplot as plt
import mpl_toolkits.mplot3d as Axes3D


# In[5]:


X,Y = make_circles(n_samples=500, noise=0.02)
print(X.shape, Y.shape)


# In[6]:


plt.scatter(X[:,0], X[:,1], c=Y)
plt.show()


# In[7]:


def phi(X):
    """Non linear transformation"""
    x1 = X[:,0]
    x2 = X[:,1]
    x3 = x1**2 + x2**2
    newX = np.zeros((X.shape[0], 3))
    newX[:,:-1] = X
    newX[:,-1] = x3
    return newX


# In[8]:


X_ = phi(X)
print(X_[:3])


# In[36]:


def plot3D(X,Y):
    fig = plt.figure(figsize=(10,10))
    ax = fig.add_subplot(111, projection='3d')
    ax.scatter(X[:,0], X[:,1], X[:,2], c=Y, depthshade=True, s=20, zdir='z')
#     plt.show()
    return ax


# In[20]:


plot3D(X_,Y)


# ## Logistic Classifier

# In[21]:


from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import cross_val_score


# In[28]:


lr = LogisticRegression(solver='lbfgs')


# In[29]:


# 5 fold cross validation
acc = cross_val_score(lr, X, Y, cv=5).mean()
print("Accuracy X(2D) is %.4f"%(acc*100))


# ## Logistic Classifier in Higher Dimensional Space

# In[30]:


acc = cross_val_score(lr, X_, Y, cv=5).mean()
print("Accuracy X(3D) is %.4f"%(acc*100))


# ## Visualize Decision Surface

# In[31]:


lr.fit(X_, Y)


# In[32]:


print(lr.coef_, lr.intercept_)


# In[35]:


# Mesh grid
xx, yy = np.meshgrid(range(-2,2), range(-2,2))
# ax + by + cz + d = 0
# z = -(ax + by + d)/c
W = lr.coef_[0]
B = lr.intercept_
z = -(W[0]*xx + W[1]*yy + B)/W[2]
print(z)


# In[39]:


ax = plot3D(X_,Y)
ax.plot_surface(xx,yy,z,alpha=0.5)
plt.show()


# In[ ]:




