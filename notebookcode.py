
# coding: utf-8

# $\newcommand{\xv}{\mathbf{x}}
# \newcommand{\Xv}{\mathbf{X}}
# \newcommand{\yv}{\mathbf{y}}
# \newcommand{\zv}{\mathbf{z}}
# \newcommand{\av}{\mathbf{a}}
# \newcommand{\Wv}{\mathbf{W}}
# \newcommand{\wv}{\mathbf{w}}
# \newcommand{\tv}{\mathbf{t}}
# \newcommand{\Tv}{\mathbf{T}}
# \newcommand{\muv}{\boldsymbol{\mu}}
# \newcommand{\sigmav}{\boldsymbol{\sigma}}
# \newcommand{\phiv}{\boldsymbol{\phi}}
# \newcommand{\Phiv}{\boldsymbol{\Phi}}
# \newcommand{\Sigmav}{\boldsymbol{\Sigma}}
# \newcommand{\Lambdav}{\boldsymbol{\Lambda}}
# \newcommand{\half}{\frac{1}{2}}
# \newcommand{\argmax}[1]{\underset{#1}{\operatorname{argmax}}}
# \newcommand{\argmin}[1]{\underset{#1}{\operatorname{argmin}}}$

# # Assignment 5: Neural Networks

# *Ben Newell*

# ## Overview

# You will write and apply code that trains neural networks of various numbers of hidden layers and units in each hidden layer and returns results as specified below.  You will do this once for a regression problem and once for a classification problem. 

# ## Required Code

# Download [nn2.tar](http://www.cs.colostate.edu/~anderson/cs440/notebooks/nn2.tar) that was used in lecture and extract its contents, which are
# 
# * `neuralnetworks.py`
# * `scaledconjugategradient.py`
# * `mlutils.py`

# Write the following functions that train and evaluate neural network models.
# 
# * `results = trainNNs(X, T, trainFraction, hiddenLayerStructures, numberRepetitions, numberIterations, classify)`
# 
# The arguments to `trainNNs` are
# 
# * `X` is a matrix of input data of shape `nSamples x nFeatures`
# * `T` is a matrix of target data of shape `nSamples x nOutputs`
# * `trainFraction` is fraction of samples to use as training data. 1-`trainFraction` is number of samples for testing data
# * `hiddenLayerStructures` is list of network architectures. For example, to test two networks, one with one hidden layer of 20 units, and one with 3 hidden layers with 5, 10, and 20 units in each layer, this argument would be `[[20], [5, 10, 20]]`.
# * `numberRepetitions` is number of times to train a neural network.  Calculate training and testing average performance (two separate averages) of this many training runs.
# * `numberIterations` is the number of iterations to run the scaled conjugate gradient algorithm when a neural network is trained.
# * `classify` is set to `True` if you are doing a classification problem, in which case `T` must be a single column of target class integers.
# 
# This function returns `results` which is list with one element for each network structure tested.  Each element is a list containing 
# 
# * the hidden layer structure (as a list),
# * a list of training data performance for each repetition, 
# * a list of testing data performance for each repetition, and
# * the number of seconds it took to run this many repetitions for this network structure.
# 
# This function should follow these steps:
# 
#   * For each network structure given in `hiddenLayerStructures`
#     * For numberRepetitions
#       * Use `ml.partition` to randomly partition X and T into training and testing sets.
#       * Create a neural network of the given structure
#       * Train it for numberIterations
#       * Use the trained network to produce outputs for the training and for the testing sets
#       * If classifying, calculate the fraction of samples incorrectly classified for training and testing sets.
#        Otherwise, calculate the RMSE of training and testing sets.
#       * Add the training and testing performance to a collection (such as a list) for this network structure
#     * Add to a collection of all results the hidden layer structure, lists of training performance and testing performance, and seconds taken to do these repetitions.
#   * return the collection of all results

# Also write the following two functions. `summarize(results)` returns a list of lists like `results` but with the list of training performances replaced by their mean and the list of testing performances replaced by their mean.   
# `bestNetwork(summary)` takes the output of `summarize(results)` and returns the best element of `results`, determined by the element that has the smallest test performance.
# 
# * `summary = summarize(results)` where `results` is returned by `trainNNs` and `summary` is like `results` with the training and testing performance lists replaced by their means
# * `best = bestNetwork(summary)` where `summary` is returned by `summarize` and `best` is the best element of `summary`

# Here are the imports from the supplied code. 

# In[1]:


import neuralnetworks as nn
import scaledconjugategradient as scg
import mlutils as ml
import time


# In[75]:


def trainNNs(X, Y, trainFraction, hiddenLayerStructures, numberRepetitions,
             numberIterations, classify = False, residuals = False):
    #In statistics and in the neural net code, the "target" matrix is Y
    #I'm more used to this.
    results = []
    start = time.time()
    for structure in hiddenLayerStructures:
        trainedList = []
        testedList = []
        for _ in range(numberRepetitions):
            #Unsure about what the validation return does.
            Xtrain,Ytrain,Xtest,Ytest = ml.partition(X, Y, 
                                        (trainFraction, 1 - trainFraction), False)
            #create nn with the right structure.
            #initiallizes ni->columns of X
            #             no->columns of Y
            #             nhs->our given structure in hidden layers
            if not classify:
                nnet = nn.NeuralNetwork(X.shape[1], structure, Y.shape[1])
            else:
                nnet = nn.NeuralNetworkClassifier(X.shape[1],structure, len(np.unique(Y)))
            #Train for number of iterations.
            nnet.train(Xtrain, Ytrain, numberIterations)
            #collect the trained outputs with training data and test data
            predictedYTrained = nnet.use(Xtrain)
            predictedYTested = nnet.use(Xtest)
            #calculate the mean standard error for these dudes.
            # VVVVVVVV THE BELOW IS FOR SIGMA SQUARED, DATA ITSELF
            # not the statistically correct way of doing it, need n - estimated paramaters df.
            # so for simple regression n -2 df, = number of columns in X with column of ones.
            #Difference between measuring the amount of error and trying to estimate
            #Epsilons in the model.
            if not classify:
                rmseTrained = np.sqrt(np.mean((Ytrain - predictedYTrained) ** 2))
                rmseTested = np.sqrt(np.mean((Ytest - predictedYTested) ** 2))
            else:
                rmseTrained = sum(Ytrain.ravel()==predictedYTrained.ravel())/ float(len(Ytrain)) * 100
                rmseTested = sum(Ytest.ravel()==predictedYTested.ravel())/ float(len(Ytest)) * 100
            #could also save the residuals here and talk about trends...
            trainedList.append(rmseTrained)
            testedList.append(rmseTested)
        end = time.time()
        results.append([structure, trainedList, testedList, end - start])
        if residuals:
            results.append([Ytrain - predictedYTrained, Ytest - predictedYTested])            
    return results


# In[3]:


def linearRegression(X,Y):
    newX = np.c_[np.ones(len(X)), X] #Adds a column of ones to the beginning for intercept.
    #This finds the OLS estimator for the model
    #beta = (XtX)-1XtY. Thanks cooley!
    betas = np.linalg.inv((np.transpose(newX) @ newX)) @ np.transpose(newX) @ Y
    # our prediction will the be Xo @ beta.
    predictedYs = newX @ betas
    rmse = np.sqrt(np.mean((predictedYs - Y) ** 2))
    #not sdhat     |msres~~~~~~~~~~~~~~~~~~~~~~~~~| <- actually not, should be adjusted for df.
    return rmse, betas


# In[4]:


def summarize(results):
    summarized = list(range(len(results)))
    for t in range(len(results)):
        summarized[t] = [results[t][0], np.mean(results[t][1]), np.mean(results[t][2]), results[t][3]]
    return summarized


# In[5]:


def bestNetwork(summary):
    lowestError = float('inf')
    lowestNetwork = summary[0]
    for s in summary:
        if s[2] < lowestError:
            lowestError = s[2]
            lowestNetwork = s
        
    return lowestNetwork


# ## Examples
# 

# In[6]:


import neuralnetworks as nn
import numpy as np

import matplotlib.pyplot as plt
get_ipython().run_line_magic('matplotlib', 'inline')


# In[7]:


X = np.arange(10).reshape((-1,1))
T = X + 1 + np.random.uniform(-1, 1, ((10,1)))


# In[8]:


Xtrain,Train, Xtest,Ttest =ml.partition(X, T, (0.7, 0.3), False)


# In[9]:


X


# In[10]:


print(Xtrain)
print(Train)
print(Xtest)
print(Ttest)
print(X)
print(T)


# In[11]:


plt.plot(X, T, 'o-');


# In[12]:


nnet = nn.NeuralNetwork(X.shape[1], 2, T.shape[1])
nnet.train(X, T, 100)
nnet.getErrorTrace()


# In[13]:


nnet = nn.NeuralNetwork(X.shape[1], [5, 5, 5], T.shape[1])
nnet.train(X, T, 100)
nnet.getErrorTrace()


# In[14]:


results = trainNNs(X, T, 0.8, [2, 10, [10, 10]], 5, 100, classify=False)
results


# In[15]:


results = trainNNs(X, T, 0.8, [0, 1, 2, 10, [10, 10], [5, 5, 5, 5], [2]*5], 50, 100, classify=False)


# In[16]:


summarize(results)


# In[17]:


model = linearRegression(X, T)


# Here we print out the model, which takes the form of RMSE, array of betas (Weights)

# In[18]:


model


# Linear regression plot of model versus line

# In[19]:


def abline(slope, intercept):
    #borrowed from: 
    #https://stackoverflow.com/questions/7941226/how-to-add-line-based-on-slope-and-intercept-in-matplotlib
    """Plot a line from slope and intercept"""
    axes = plt.gca()
    x_vals = np.array(axes.get_xlim())
    y_vals = intercept + slope * x_vals
    plt.plot(x_vals, y_vals, '--')

plt.plot(X, T, ".")
abline(model[1][1], model[1][0])


# In[20]:


best = bestNetwork(summarize(results))
print(best)
print('Hidden Layers {} Average RMSE Training {:.2f} Testing {:.2f} Took {:.2f} seconds'.format(*best))


# Hummm...neural nets with no hidden layers did best on this simple data set.  Why?  Remember what "best" means.

# Because its a linear trend!

# ## Data for Regression Experiment
# 
# From the UCI Machine Learning Repository, download the [Appliances energy prediction](http://archive.ics.uci.edu/ml/datasets/Appliances+energy+prediction) data.  You can do this by visiting the Data Folder for this data set, or just do this:
# 
#      !wget http://archive.ics.uci.edu/ml/machine-learning-databases/00374/energydata_complete.csv
# 
# 

# Read this data into python.  One suggestion is to use the `loadtxt` function in the `numpy` package.  You may ignore the first column of each row which contains a data and time.  Also ignore the last two columns of random variables.  We will not use that in our modeling of this data.  You will also have to deal with the double quotes that surround every value in every field.  Read the first line of this file to get the names of the features.
# 
# Once you have read this in correctly, you should see values like this:

# CSV reading

# In[21]:


import pandas as pd
csv = pd.read_csv("energydata_complete.csv")
#drop unneeded columns
csv = csv.drop(['date', 'rv1', 'rv2'], axis = 1)


# In[22]:


#names
csv.columns


# In[23]:


data = csv.values #.values is a numpy array! 
data.shape


# In[24]:


data


# In[25]:


data[:2,:]


# Use the first two columns, labelled `Appliances` and `lights` as the target variables, and the remaining 24 columns as the input features.  So

# In[26]:


Tenergy = data[:, :2]
Xenergy = data[:, 2:]


# In[27]:


type(Xenergy[1])


# In[28]:


Xenergy.shape, Tenergy.shape


# If you were stuck on a desert island with a raspberry pi, this might be the fastest way to go. 

# In[29]:


mod = linearRegression(Xenergy, Tenergy)


# In[30]:


mod


# Train several neural networks on all of this data for 100 iterations.  Plot the error trace (nnet.getErrorTrace()) to help you decide now many iterations might be needed.  100 may not be enough.  If for your larger networks the error is still decreasing after 100 iterations you should train all nets for more than 100 iterations.
# 
# Now use your `trainNNs`, `summarize`, and `bestNetwork` functions on this data to investigate various network sizes.

# In[31]:


results = trainNNs(Xenergy, Tenergy, 0.8, [0, 5, [5, 5], [10, 10]], 10, 100)


# In[32]:


summarize(results)


# In[33]:


bestNetwork(summarize(results))


# Here, plot the residuals of the best versus the residuals of the linear regression to see what sort of trend neural net might have picked up. 

# Test at least 10 different hidden layer structures.  Larger numbers of layers and units may do the best on training data, but not on testing data. Why?
# 
# Now train another network with your best hidden layer structure on 0.8 of the data and use the trained network on the testing data (the remaining 0.2 of the date).  As before use `ml.partition` to produce the training and testing sets.
# 
# For the testing data, plot the predicted and actual `Appliances` energy use, and the predicted and actual `lights` energy use, in two separate plots.  Discuss what you see.

# Create new nn with .8 for training and .2 for testing. Best was 10,10

# In[40]:


results = trainNNs(Xenergy, Tenergy, 0.8, [[1,1,1,1,1,1], [5, 5, 5], [20, 20], [10, 10, 10, 10],[100]], 10, 100)


# In[35]:


restultsPlus = trainNNs(Xenergy, Tenergy, 0.8, [[100, 100, 100], [20,20,20,20,20], [1000]], 10, 100)


# In[41]:


summarize(results)


# In[42]:


summarize(restultsPlus)


# In[44]:


#Create and train the nn.
#Do this with 1000, but maybe on your desktop.
bestNN = nn.NeuralNetwork(Xenergy.shape[1], [10,10], Tenergy.shape[1])
Xtrain,Ytrain,Xtest,Ytest=ml.partition(Xenergy, Tenergy, (.8, .2), False)
bestNN.train(Xtrain, Ytrain, 350) #300 should be fine
#Final error is not the same as RMSE


# In[45]:


plt.plot(bestNN.getErrorTrace())


# In[46]:


Ypredicted = bestNN.use(Xenergy)


# In[47]:


plt.plot(Tenergy[:,0], Ypredicted[:,0], ".")


# Here is a plot of actual vs predicted. Really doesn't look great in terms of predictive power

# In[48]:


plt.plot(Tenergy[:,1], Ypredicted[:,1], ".")


# In[49]:


np.unique(Tenergy[:,1])


# Here it is for the other one, not appliance maybe?

# Here are some plots of residuals!

# In[50]:


plt.plot(Tenergy[:,0] - Ypredicted[:,0], ".")


# In[51]:


plt.plot(Tenergy[:,1] - Ypredicted[:,1], ".")


# ## Data for Classification Experiment
# 
# From the UCI Machine Learning Repository, download the [Anuran Calls (MFCCs)](http://archive.ics.uci.edu/ml/datasets/Anuran+Calls+%28MFCCs%29) data.  You can do this by visiting the Data Folder for this data set, or just do this:
# 
#      !wget 'http://archive.ics.uci.edu/ml/machine-learning-databases/00406/Anuran Calls (MFCCs).zip'
#      !unzip Anuran*zip
#      
# Read the data in the file `Frogs_MFCCs.csv` into python.  This will be a little tricky. Each line of the file is a sample of audio features plus three columns that label the sample by family, genus, and species. We will try to predict the species.  The tricky part is that the species is given as text.  We need to convert this to a target class, as an integer. The `numpy` function `unique` will come in handy here.

# In[52]:


csv = pd.read_csv("Frogs_MFCCs.csv")
csv = csv.drop(['RecordID', 'Family', 'Genus', 'MFCCs_ 1'], axis = 1) #also drops MFCCs_1, 
                                                                    #which only has the value 1, no in his.


# In[53]:


csv.shape


# In[54]:


data = csv.values
Tanuran = data[:, 21:]
Xanuran = data[:, :21]
Xanuran = Xanuran.astype(np.float64)
#print(type(Xanuran[1,1]))
#print(Xanuran.astype(np.float64))
#print(type(Xenergy[1,1]))


# In[55]:


Xanuran.shape, Tanuran.shape


# In[56]:


Xanuran[:2,:]


# In[57]:


names = np.unique(Tanuran)
names = list(names)
for i in range(len(Tanuran)):
    Tanuran[i] = names.index(Tanuran[i])
#Tanuran = [names.index(n) for n in Tanuran]
#Tanuran = [[x] for x in Tanuran]
#Tanuran = np.array(Tanuran)


# In[65]:


Xanuran.shape[1]


# In[62]:


Tanuran = Tanuran.astype(np.int)
Tanuran[:2]


# In[63]:


for i in range(10):
    print('{} samples in class {}'.format(np.sum(Tanuran==i), i))


# In[76]:


results = trainNNs(Xanuran, Tanuran, 0.8, [0, 5, [5, 5]], 5, 100, classify=True)


# In[77]:


summarize(results)


# In[27]:


bestNetwork(summarize(results))


# Now do an investigation like you did for the regression data. 
# 
# Test at least 10 different hidden layer structures. Then train another network with your best hidden layer structure on 0.8 of the data and use the trained network on the testing data (the remaining 0.2 of the date). 
# 
# Plot the predicted and actual `Species` for the testing data as an integer.  Discuss what you see.

# ## Grading
# 
# Download [A5grader.tar](http://www.cs.colostate.edu/~anderson/cs440/notebooks/A5grader.tar) and extract `A5grader.py` from it.

# In[114]:


get_ipython().run_line_magic('run', '-i "A5grader.py"')


# ## Check-in

# Do not include this section in your notebook.
# 
# Name your notebook ```Lastname-A5.ipynb```.  So, for me it would be ```Anderson-A5.ipynb```.  Submit the file using the ```Assignment 5``` link on [Canvas](https://colostate.instructure.com/courses/68135).

# ## Extra Credit
# 
#   2. Repeat the above regression and classification experiments with a second regression data set and a second classification data set.
#   
#   2. Since you are collecting the performance of all repetitions for each network structure, you can calculate a confidence interval about the mean, to help judge significant differences. Do this for either the regression or the classification data and plot the mean test performance with confidence intervals for each network structure tested.  Discuss the statistical significance of the differences among the means.  One website I found to help with this is the site [Correct way to obtain confidence interval with scipy](https://stackoverflow.com/questions/28242593/correct-way-to-obtain-confidence-interval-with-scipy).
#   
# 
