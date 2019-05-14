
# coding: utf-8

# # Assignment 5
# # Neural Network
# submitted by <br>
# roll no : 2018201051
# 

# ## Question 1 : Creating Neural Network 

# In[ ]:

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


# In[6]:

# from google.colab import files
# uploaded = files.upload()

# from google.colab import files
# uploaded = files.upload()

get_ipython().system(u'unzip data.zip')


# In[ ]:

def sigmoid(Z):
    temp = 1 + np.exp(-Z)
    return 1./temp


# In[ ]:

def relu(Z):
    temp  = np.where(Z >= 0 , 1 , 0)
    return Z*temp


# In[ ]:

def softmax(Z):
    temp = np.exp(Z)
    temp1 = np.sum(temp , axis = 0)
    return np.divide(temp , temp1 , dtype = "float")


# In[ ]:

def y_transform(y):
    arr =  []
    for i  in range(y.shape[0]):
        temp = [0,0,0,0,0,0,0,0,0,0]
        val = y.iloc[i]
        temp[val] = 1 
        arr.append(temp)
    arr = np.array(arr)
    return arr.T


# In[ ]:

def linear_backward( dZ , A_prev , W):
    m = A_prev.shape[1]
    dW = np.dot(dZ, A_prev.T)/m
    db = np.sum(dZ, axis = 1, keepdims = True)/m
    dA_prev = np.dot(W.T, dZ)
    return dA_prev , db , dW

def relu_backward( dA , Z):
    temp = np.where(Z >= 0 , 1 ,0)
    return dA*temp

def sig_backward( dA , Z):
    temp  = sigmoid(Z)*(1-sigmoid(Z))
    return dA*temp
        
def tanh_backward( dA , Z):
    temp = 1.0 - np.tanh(Z)**2
    return dA*temp


# In[ ]:

class neuralNet():
    def __init__(self,  dims = [] , fun = [] ,iterations = 30, learning_rate = 0.1):# dims  = [input_dims,  h1layer , .... , out_layer] , fun = activation function at each layer
        self.num_iters = iterations
        self.num_layers = len(dims)-1
        self.parameters = {}
        self.learning_rate = learning_rate
        self.training_error = []
        self.val_error = []
        
        for i in range(1, self.num_layers +1):
            self.parameters["W" + str(i)] = np.random.randn(dims[i],dims[i-1])
            self.parameters["b" + str(i)] = np.zeros((dims[i] , 1))
            self.parameters["activation" + str(i)] = fun[i-1]
    
    def train_forward(self,X) :
        caches  = []
        A_old  =  X
        for i  in range(1, self.num_layers + 1) :

            temp  = np.matmul(self.parameters["W" + str(i)], A_old) 
            Z  = temp  +  self.parameters["b" + str(i)]
            
            if  np.isnan(np.sum(Z)) :
                print("Error at Layer :  " +  str(i))
                print(Z)

            if self.parameters["activation" + str(i)] == "relu" :
                A_new = relu(Z)
      
            elif self.parameters["activation" + str(i)] == "tanh" :
                A_new = np.tanh(Z)


            elif self.parameters["activation" + str(i)] == "sigmoid":
                A_new = sigmoid(Z)

            elif self.parameters["activation" + str(i)] == "softmax":
                A_new = softmax(Z)
            
            cache = (A_old, Z)
            caches.append(cache)
            A_old = A_new
        return A_new , caches
    
    
        
    def train_backward(self , Al , y , caches):
        grads = {}
        L = self.num_layers
        A_prev  , Z  = caches[ L - 1 ]
        dZl = Al - y  # for softmax layer only
        dA_prev , db , dW  = linear_backward( dZl , A_prev  ,self.parameters["W" + str(L)])
#         grads["dA" + str(L-1)] = dA_prev
        grads["dW" + str(L)] = dW
        grads["db" + str(L)] = db
        for  i in range(L-1,0,-1) :
            A_prev , Z = caches[i-1]
            if self.parameters["activation" + str(i)] == "relu":
                dZ = relu_backward(dA_prev , Z)
            elif self.parameters["activation" + str(i)] == "sigmoid":
                dZ = sig_backward(dA_prev,  Z)
            elif self.parameters["activation" + str(i)] == "tanh":
                dZ = tanh_backward(dA_prev,  Z)
            dA_prev, db , dW = linear_backward(dZ ,A_prev , self.parameters["W" + str(i)])

            grads["dW" + str(i)] = dW
            grads["db" + str(i)] = db
        return grads  
            
        
    def update_weights(self, grads):
        for l in range(self.num_layers):

            self.parameters["W" + str(l+1)] += -self.learning_rate*grads["dW" + str(l+1)]
            self.parameters["b" + str(l+1)] += -self.learning_rate*grads["db" + str(l+1)]

    def fit(self , X, Y ,val_X , val_Y,  batch_size = 5000):
        l = X.shape[0]/batch_size
        
        val_X = val_X.astype('float')/255
        val_Y = y_transform(val_Y)
        
        for  i in range(self.num_iters):
            print("Epoch ========== " + str(i)+" :=========== ")
            cost = 0
            for j in range(l):
                x_batch =  X.iloc[j*batch_size : (j+1)*batch_size , :].astype('float')
                y_batch =  Y.iloc[j*batch_size : (j+1)*batch_size ]
                x_batch = x_batch/255

                Al , caches = self.train_forward(x_batch.T)
                y_batch  = y_transform(y_batch)
                cost += self.error(Al , y_batch) # y-batch should be ( nl * m )
                grads = self.train_backward(Al ,y_batch , caches)
                self.update_weights(grads)
            
            self.training_error.append(float(cost)/X.shape[0])
            
            #Calculating validation error
            al_val, _ = self.train_forward(val_X.T)
            val_e  = float(self.error(al_val,val_Y))/val_X.shape[0]
            self.val_error.append(val_e)
            print("training error is  " + str(float(cost)/X.shape[0]))
            print("val error is : " + str(val_e))
    
    def error(self, al ,  y ):
        n_max =  np.max(al  ,axis = 0 )
        result =  np.where(al >=n_max , 1 , 0)
        cost  = np.sum(np.sum(np.abs(result-y) , axis  = 0)/2)
        return float(cost)
    
    def epoch_vs_accuracy(self):
        plt.plot( range(self.num_iters), self.training_error , 'r')
        plt.plot(range(self.num_iters) , self.val_error , 'g')
        plt.xlabel("numbers of Epochs")
        plt.ylabel("error")
        plt.legend(["training error" , "validation error"])
        plt.show()
        
    
    def predict(self , X):
        X = X.astype('float')/255
#         y = y_transform(y)
        al, caches = self.train_forward(X.T)
        n_max =  np.max(al  ,axis = 0)
        result =  np.where(al >=n_max , 1 , 0)
        temp =  np.array([0,1,2,3,4,5,6,7,8,9]).reshape(10,1)
        n_max2 = np.max(result*temp,axis = 0)
        return n_max2
                
#         


# ##  Loading Apparel dataset

# In[ ]:

data = pd.read_csv('data.csv')
data.head()
train_data = data.iloc[:50000,:]
val_data = data.iloc[50000:,:]


# In[ ]:

train_X = train_data.iloc[: , 1:]
train_Y = train_data.iloc[: , 0]
val_X = val_data.iloc[:,1:]
val_Y = val_data.iloc[:,0]


# ## Some examples 

# ###  Training a neural network with 2 hidden layers and sigmoid function   

# In[ ]:

NN = neuralNet([784 ,300, 100 , 10] ,['sigmoid', 'sigmoid' , 'softmax'] , iterations = 50)


# In[16]:

NN.fit(train_X ,train_Y ,val_X , val_Y  , 500)


# ###  Predicting on test data

# In[ ]:

test_data = pd.read_csv("apparel-test (1).csv")


# In[25]:

test_result = NN.predict(test_data)
print(test_result)
np.savetxt("foo.csv", test_result , fmt = "%d")

files.download('foo.csv')


# In[19]:

NN.epoch_vs_accuracy()


# ### Training with relu function 

# In[ ]:

NN1 = neuralNet([784 ,64, 10] ,['relu' , 'softmax'] , iterations = 50, learning_rate = 0.005)


# In[23]:

NN1.fit(train_X , train_Y,val_X , val_Y , 100)


# In[24]:

NN1.epoch_vs_accuracy()


# ###  Training with Tanh function( 2 hidden layers )

# In[ ]:

NN2 = neuralNet([784 ,256 , 256, 10] ,['tanh' ,'tanh' , 'softmax'] , iterations = 50, learning_rate = 0.1)


# In[27]:

NN2.fit(train_X , train_Y , val_X ,val_Y , 100)


# In[28]:

NN2.epoch_vs_accuracy()


# ### Training with Sigmoid 

# In[ ]:

NN3 = neuralNet([784 ,1024, 10] ,['sigmoid' , 'softmax'] , iterations = 50 , learning_rate = 0.1)


# In[30]:

NN3.fit(train_X,  train_Y , val_X , val_Y , 100)


# In[32]:

NN3.epoch_vs_accuracy()


# ## Accuracy variation with no of layers :  

# In[34]:

layers =  [1,2,3]
error_rates =  []

# 1 hidden layer
NN4 = neuralNet([784 ,256, 10] ,['sigmoid' , 'softmax'] , iterations = 20 , learning_rate = 0.1)
NN4.fit(train_X,train_Y , val_X , val_Y , 500)
error_rates.append(NN4.val_error[-1])

# 2 hidden layer
NN5 = neuralNet([784 ,256,256, 10] ,['sigmoid',  'sigmoid' , 'softmax'] , iterations = 20 , learning_rate = 0.1)
NN5.fit(train_X , train_Y , val_X , val_Y , 500)
error_rates.append(NN5.val_error[-1])

#3 hidden layers

NN6 = neuralNet([784 ,256,256,256, 10] ,['sigmoid','sigmoid','sigmoid','softmax'] , iterations = 20 , learning_rate = 0.1)
NN6.fit(train_X , train_Y , val_X , val_Y , 500)
error_rates.append(NN6.val_error[-1])


plt.plot(layers , error_rates , 'g')
plt.show()


# In[37]:

NN4.epoch_vs_accuracy()
NN5.epoch_vs_accuracy()
NN6.epoch_vs_accuracy()
plt.plot(layers , error_rates , 'g')
plt.xlabel("No of layers")
plt.ylabel("Error")
plt.show()


# ## Question 2 :  

# ### Requirments for the data set of house price prediction :  

# ```
# 1 . only one node will be enough in the output layer since it is a regression problem .
# 2 . Required linear activation function in the output layer(need to modifiy as softmax is used in above 
#     neural network) .
# 3 . All hidden layers should not have linear activation function since it will be unable to learn 
#     nonlinearity in the data because 3 hidden layers with all linear activation will work the same way 
#     as a single layer neural network. 
# 4 . cost function should also be changed.(ex : least mean square can be used. )
# ```
