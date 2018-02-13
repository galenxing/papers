
# coding: utf-8

# In[34]:

import scipy.io as sio
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import h5py


# In[35]:

def remove_features(X_train, X_test, indices):
    indices = np.array(indices, dtype = np.int64)
    X_train = X_train[:,indices]
    X_test = X_test[:,indices ]
    #print(X_train.shape)
    #print(X_test.shape)

    return X_train, X_test


# In[36]:

def normalize(X):
    normed = (X - X.mean(axis =0)) / X.std(axis =0)
    return normed


# In[37]:

def calculate_variances(X, n_features):
    summation = np.zeros((1,784), dtype = np.float64)
    n = len(X)

    #calculates MLE mean
    for i, value in enumerate(X):
        temp = np.array(value, dtype=np.float64)
        summation = np.add(summation,temp)  

    summation = np.divide(summation, n)
    mle_mean = summation

    #calculated cov matrix
    print("calculating cov_matrix")
    summation = np.zeros((784,784), dtype = np.float64)
    for i, value in enumerate(X):
        diff = np.subtract(value, mle_mean)
        diff = np.reshape(diff, (1, 784))
        t_diff = np.transpose(diff)
        product = np.matmul(t_diff, diff)
        summation = np.add(summation, product)

    summation = np.divide(summation, n)
    cov_matrix = summation

    total_features = len(cov_matrix)

    indices = np.arange(total_features)
    variances = []

    for i in range(total_features):
        variances.append(cov_matrix[i,i])

    variances = np.array(variances, dtype= np.float64)
    variances = np.reshape(variances, (784,1))
    indices = np.reshape(indices, (784,1))

    #print(variances.shape)
    #print(indices.shape)

    v_and_i = np.concatenate((variances, indices), axis =1)
    #print(v_and_i[:20])
    v_and_i = v_and_i[v_and_i[:,0].argsort()][::-1]
    #print()
    #print(v_and_i[:20])

    return v_and_i[:n_features,1]


# In[38]:

def preprocessing(X_train, X_test, n_features):
    indices = calculate_variances(X_train,n_features)
    X_train, X_test = remove_features(X_train, X_test, indices)
    #X_train = normalize(X_train)
    #X_test = normalize(X_test)
    return X_train, X_test


# In[39]:

def split(X,Y,split):
    full_data = np.concatenate((X,Y),axis= 1)

    np.random.shuffle(full_data)

    l = full_data.shape[0]
    train= full_data[: int(l * split)]
    test = full_data[int(l * split):]

    
    
    x_train = train[:,0:784]
    y_train = train[:,784:785]
    x_test = test[:,0:784]
    y_test = test[:,784:785]
    
    return (x_train,y_train,x_test,y_test)


# In[40]:

class Tree(object):
    
    #Each node in the tree will contain:
    #the subset of samples in the cell, the labels of those rows, and the feature/threshold that decides its children
    def __init__(self):
        self.left = None 
        self.right = None
        self.x_data = None
        self.y_data = None
        self.feature = None
        self.threshold = None
        self.leaf = None
        self.label = None
    
    def is_leaf(self):
        if self.left is None and self.right is None:
            self.leaf = True
            return True
        else:
            self.leaf = False 
            return False


# In[41]:

#Y represents the labels of all points in a cell
def classification_error(y):
    #print(y)
    label_counts = {}
    for i in range (0,y.shape[0]):
        #print(y[i][0])
        label = (y[i][0],)
        if label not in label_counts:
            label_counts[label] = 1
        else:
            label_counts[label] += 1
    
    #Determining which label was seen the most
    most_frequent_label_count = 0
    actual_most_frequent_label = None
    
    for label in label_counts:
        if label_counts[label] > most_frequent_label_count:
            most_frequent_label_count = label_counts[label]
            actual_most_frequent_label = label
    #print(actual_most_frequent_label)
    #print(most_frequent_label_count)

    if(y.shape[0] == 0): #Indicates no samples in space
        return 0
    
    classification_error = 1 - float(most_frequent_label_count/(y.shape[0]))
    #print(classification_error)
    return classification_error


# In[42]:

#Determines the feature and threshold to be used
# x_train has all top 200 feature 

def feature_threshold_determiner(x_train,y_train):     
    optimal_classification_error = -1000
    optimal_feature = None
    optimal_threshold = None
    
    left_cut_x = None
    left_cut_y = None
    right_cut_x = None
    right_cut_y = None
    
    
    #For each feature
    for feature_index in range(1,x_train.shape[1]):
        #print(x_train.shape[1])
        #For possible thresholds
        for threshold in range (64,193,64):
            
            LC_x = np.zeros((1, x_train.shape[1])) 
            LC_y = np.zeros((1, y_train.shape[1]))
    
            RC_x = np.zeros((1, x_train.shape[1]))
            RC_y = np.zeros((1, y_train.shape[1]))
            
            
            #Iterate through each sample, if feature value<threshold then sample label is in left cell
            for row in range(0, x_train.shape[0]): 
                #print('start')

                #print(x_train[row][feature_index])
                #print('end')

                if (x_train[row][feature_index] <= threshold):
                    #print("Shape: " + str(LC_x.shape))
                    x_row = np.reshape(x_train[row],(1,x_train.shape[1]))
                    y_row = np.reshape(y_train[row],(1,y_train.shape[1]))
        
                    LC_x = np.vstack((LC_x,x_row))
                    LC_y = np.vstack((LC_y,y_row))
                    
                else:
                    x_row = np.reshape(x_train[row],(1,x_train.shape[1]))
                    y_row = np.reshape(y_train[row],(1,y_train.shape[1]))
        
        
                    RC_x = np.vstack((RC_x,x_row))
                    RC_y = np.vstack((RC_y,y_row))
        
            #Removing 0 vector at start
            LC_x = LC_x[1:,:]
            LC_y = LC_y[1:,:]
            RC_x = RC_x[1:,:]
            RC_y = RC_y[1:,:]

            #print('here')
            #print(LC_x)
            #print(RC_x)
            
            total_cell_classification_error = classification_error(y_train)
            
            left_cell_classification_error = classification_error(LC_y)
            right_cell_classification_error = classification_error(RC_y)

            #print(total_cell_classification_error)
            #print(left_cell_classification_error)           
            #print(right_cell_classification_error)
            
            weighted_left_cell_error = left_cell_classification_error*LC_y.shape[0]/y_train.shape[0]
            weighted_right_cell_error = right_cell_classification_error*RC_y.shape[0]/y_train.shape[0]
            
            this_threshold_feature_error = total_cell_classification_error-weighted_left_cell_error-weighted_right_cell_error
            

            if this_threshold_feature_error > optimal_classification_error:
                #print(this_threshold_feature_error)

                optimal_classification_error = this_threshold_feature_error
                optimal_feature = feature_index
                optimal_threshold = threshold
                left_cut_x = LC_x
                left_cut_y = LC_y
                right_cut_x = RC_x
                right_cut_y = RC_y
                
    
    return optimal_feature,optimal_threshold,left_cut_x,left_cut_y,right_cut_x,right_cut_y 


# In[43]:

def add_decision(node):
    x_train = node.x_data
    y_train = node.y_data
    
    optimal_feature, optimal_threshold,LC_x,LC_y,RC_x,RC_y = feature_threshold_determiner(x_train,y_train)
  
    node.threshold = optimal_threshold #Should be value between 0,255
    node.feature = optimal_feature #Should be index
    node.left = Tree()
    node.left.x_data = LC_x
    node.left.y_data = LC_y
    node.right = Tree()
    node.right.x_data = RC_x
    node.right.y_data = RC_y
    
    return node


# In[44]:

def print_Preorder(root):
    #print("ping")
    if root:
 
        # First print the data of node
        print("Feature: " + str(root.feature) + " Threshold: " + str(root.threshold) + 
             " X Samples: " + str(root.x_data.shape) + " Y Samples: " + str(root.y_data.shape) 
              + " X_Label:" + str(root.label) + " Leaf?: " + str(root.leaf))
 
        # Then recur on left child
        print_Preorder(root.left)
 
        # Finally recur on right child
        print_Preorder(root.right)


# In[45]:

def calc_label(y):
    label_counts = {}
    for i in range (0,y.shape[0]):
        label = (y[i][0],)
        if label not in label_counts:
            label_counts[label] = 1
        else:
            label_counts[label] += 1
    
    #Determining which label was seen the most
    most_frequent_label_count = 0
    actual_most_frequent_label = None
    
    for label in label_counts:
        if label_counts[label] > most_frequent_label_count:
            most_frequent_label_count = label_counts[label]
            actual_most_frequent_label = label
    
    return actual_most_frequent_label[0]


# In[46]:

def label_tree(root):
    #print("ping")
    #print(root)
    if root is not None:
        if root.is_leaf():
            root.label = calc_label(root.y_data)
        
        else:
            # Then recur on left child
            label_tree(root.left)
 
            # Finally recur on right child
            label_tree(root.right)
    
    return root




# In[47]:

def DT_classify(root,test_point):
    if root is not None:
        if root.is_leaf():
            return root.label
        else:
            feature_index = root.feature
            threshold = root.threshold
            #print(test_point[feature_index])
            if test_point[feature_index] <= threshold:
                return DT_classify(root.left,test_point)
            else:
                return DT_classify(root.right,test_point)


# In[48]:

def correct_classification(root):
    y = root.y_data
    
    label_counts = {}
    for i in range (0,y.shape[0]):
        label = (y[i][0],)
        if label not in label_counts:
            label_counts[label] = 1
        else:
            label_counts[label] += 1
    
    #Determining which label was seen the most
    most_frequent_label_count = 0
    actual_most_frequent_label = None
    
    for label in label_counts:
        if label_counts[label] > most_frequent_label_count:
            most_frequent_label_count = label_counts[label]
            actual_most_frequent_label = label
    
    return most_frequent_label_count
    


# In[49]:

def train_error(root):
    correct_classification_count = 0
    if root is not None:
        if (root.is_leaf()):
            correct_classification_count += correct_classification(root)
        else:
            correct_classification_count += train_error(root.left)
            correct_classification_count += train_error(root.right)
    
        return correct_classification_count


# In[50]:

def DT_test_accuracy(x_test,y_test,root):
    
    correct_classification = 0
    for i in range(0,x_test.shape[0]):
        if DT_classify(root,x_test[i]) == y_test[i][0]:
            #print("ping")
            correct_classification += 1
        
    test_accuracy = float(correct_classification/x_test.shape[0])    
    
    test_error = float(1-test_accuracy) 
    
    return test_error
    
    


# In[56]:

#Find train and test error as we go through the and add more decisions
def train_test_error(x_train, y_train, x_test,y_test,k):
    root = Tree()
    root.x_data = x_train
    root.y_data = y_train
    
    k_values = []
    train_errors = []
    test_errors = []
    
    queue = []
    queue.append(root)
    
    i = 0
    
    while i<k:
        print(i)
        add_decision(queue[i])
        queue.append(queue[i].left)
        queue.append(queue[i].right)
        i += 1
        
        k_values.append(i)
        
        label_tree(root)
        
        training_set_error = 1-(train_error(root)/x_train.shape[0])
        train_errors.append(training_set_error)
        
        test_error = DT_test_accuracy(x_test,y_test,root)
        test_errors.append(test_error)
    
    return k_values,train_errors,test_errors


# In[57]:

def main():
    mat_content = sio.loadmat("hw1data.mat")
    X = np.array(mat_content['X'],dtype = np.float64)
    Y = np.array(mat_content['Y'],dtype = np.float64)

    X_modified = X 
    Y_modified = Y
    
    x_train,y_train,x_test,y_test = split(X_modified,Y_modified,.7)
    
    n_features = 200

    x_train, x_test = preprocessing(x_train, x_test, n_features)

    
    
    k_values,train_errors,test_errors = train_test_error(x_train,y_train,x_test,y_test,10)
    
    print(k_values)
    print(train_errors)
    print(test_errors)

    



# In[55]:

main()


# In[ ]:




# In[ ]:




# In[ ]:



