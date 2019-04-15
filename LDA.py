#!/usr/bin/env python
# coding: utf-8

# In[1]:


import sklearn as sk
import numpy as np
from numpy.linalg import inv
from zipfile import ZipFile
from sklearn import metrics


# In[2]:


file = ZipFile('orl_faces.zip') 
file.extractall() 
file.close()


# In[3]:


from matplotlib import pyplot as py
folder = 1 
file = 1 
data = [] 
#generate data matrix 
for i in range(1,41):    
    for j in range(1,11):        
        data.append(py.imread("orl_faces/"+"s" + str(i) + "/"+str(j )+".pgm"))   #append data from the folder path


# In[4]:


#py.imshow(data[20])


# In[5]:


#convert the images into vectors
for i in range(0,400):
    data[i] = data[i].reshape(10304)


# In[6]:


#Stack all the vectors into a single data matrix D
D = np.array(data)


# In[7]:


#Construct labels
#Generate label vector Y
labels = []
for i in range(1,41):
    for j in range(1,11):
        labels.append(i)
        
Y = np.array(labels)


# In[8]:


#Split dataset
#Splitting Data to training set and testing set 
D_train = D[list(range(0,400,2))] 
Y_train = Y[list(range(0,400,2))]
D_test = D[list(range(1,400,2))]
Y_test = Y[list(range(1,400,2))]


# # LDA

# In[9]:


#Class-specific subsets and class means
mean_vector = list()
for i in range(0,200,5):
    mean = np.mean(D_train[[i, i+1, i+2, i+3, i+4], :], axis=0)
    mean_vector.append(mean)
mean_vector = np.array(mean_vector)
overall_mean = np.mean(mean_vector, axis=0)


# In[34]:


#Between class scatter matrix
Sb = 0
for i in range(0,40):
    Sb = Sb + 5 * np.dot((mean_vector[i]-overall_mean), np.transpose(mean_vector[i]-overall_mean))


# In[23]:


#center class matrices
class_vector = list()
centered_matrices = list()
for i in range(0,200,5):
    classes = D_train[[i, i+1, i+2, i+3, i+4], :]
    class_vector.append(classes)
class_vector = np.array(class_vector)
for j in range(0,40):
    centered = class_vector[j] - mean_vector[j]
    centered_matrices.append(centered)
centered_matrices = np.array(centered_matrices)


# In[31]:


#Within-class scatter matrix
S = 0
for i in range(0,40):
    S = S + np.dot(np.transpose(centered_matrices[i]), centered_matrices[i])


# In[ ]:


#Dominant Eigenvector
Sinv = inv(S)
eigvals, eigvec = np.linalg.eigh(np.dot(Sinv,Sb))


# In[ ]:


#Using 39 dominant eigenvectors
dominant_eigvals = list()
dominant_eigvec = list()
eigPairs = [(np.abs(eigvals[i]), eigvec[:, i]) for i in range(len(eigvals))]
eigPairs.sort(key = lambda x:x[0], reverse = True)
sortedeigvals = sorted(eigvals, reverse = True)
for i in range(0,39):
    dominant = sortedeigvals[i]
    vec = eigPairs[i][1]
    dominant_eigvec.append(vec)
dominant_eigvec = np.array(dominant_eigvec)


# In[ ]:


#Projection of the training and test sets
Training_set_Transform = np.dot(D_train, np.transpose(dominant_eigvec))
Test_set_Transform = np.dot(D_test, np.transpose(dominant_eigvec))


# In[ ]:


#KNN
# instantiate learning model 
knn = KNeighborsClassifier(n_neighbors=1)

# fitting the model
knn.fit(Training_set_Transform,Y_train)

# predict the response
pred = knn.predict(Test_set_Transform)


# In[ ]:


# Model Accuracy, how often is the classifier correct?
print("Accuracy:",metrics.accuracy_score(Y_test,pred))

