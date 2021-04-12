# -*- coding: utf-8 -*-
"""
Spyder Editor

This is a temporary script file.
"""

# ------------>>>>>>>> RUN THIS CODE CELL <<<<<<<<------------
# === CELL TYPE: IMPORTS AND SETUP 

import sys                       # for testing use only
import os                        # for testing use only
from datetime import datetime    # for testing use only
import random                    # for testing use only
import hashlib                   # for testing use only
import pandas as pd
import numpy as np
import math 
import statistics


# 1.a.
# ------------>>>>>>>> RUN THIS CODE CELL <<<<<<<<------------
# --------  (run after implementation)
# === CODE TYPE: ANSWER 

def myName():
    # YOUR CODE HERE
    return 'Roi Susi'



# 1.a.
# ------------>>>>>>>> RUN THIS CODE CELL <<<<<<<<------------
# --------  (run after implementation)
# === CODE TYPE: SELF TESTING
# Use the following code to test your implementation:
student_name = myName()
# --- add additional code to check your code if needed:
# YOUR CODE HERE


# 1.b.
# ------------>>>>>>>> RUN THIS CODE CELL <<<<<<<<------------
# --------  (run after implementation)
# === CODE TYPE: ANSWER 

def myId():
    # YOUR CODE HERE
    return 300685906


# 1.b.
# ------------>>>>>>>> RUN THIS CODE CELL <<<<<<<<------------
# --------  (run after implementation)
# === CODE TYPE: SELF TESTING
# Use the following code to test your implementation:
student_id = myId()
# --- add additional code to check your code if needed:
# YOUR CODE HERE



# 1.a. 1.b.
# ------------>>>>>>>> RUN THIS CODE CELL <<<<<<<<------------
# --------  (run only)
# === CODE TYPE: GRADED TEST 

print ("Part 1.a. 1.b. - Test 1 (name: test1-1_student_info, points: 0.5)")
print ("\t--->Testing the implementation of 'myName' and 'myId' ...")

# dataframe for output:
dt1 = datetime.now()
try:
    student_name, student_id = myName(), myId()
except Exception as e:
    print ('You probably have a syntax or implementation error,  \nerror Message:',str(e), '\nTry fixing your code')
    raise

assert type(student_name) is str or type(student_id) is int, "name is not a string or id is not an integer"

s_datetime =  [dt1.strftime('%Y-%m-%d %H:%M:%S'), datetime.now().strftime('%Y-%m-%d %H:%M:%S')]
df_output = pd.DataFrame({'value': [student_name, student_id], 'type': [type(student_name), type(student_id)], 'date_time': s_datetime},index=['student_name','student_id'])

print ("\nGood Job!\nYou've passed the test for the implementation of 'myName' and 'myId'  :-)")

print ('\nOutput dataframe:')
print ('-----------------')
df_output



# 1.c.
# ------------>>>>>>>> RUN THIS CODE CELL <<<<<<<<------------
# === CELL TYPE: IMPLEMENTATION TASKS

# --------------------------- RUN THIS CODE CELL (IMPLEMENTATION INFO) -------------------------------------
# Run the following to get information about your implementation
# ---------------------------
def get_assignment_params(student_id):
    num_modulo = 2 ** 31-1
    created_1st_state = int(hashlib.md5(str(student_id).encode('utf-8')).hexdigest(),16) % num_modulo
    created_2nd_state = int(hashlib.md5(str(created_1st_state).encode('utf-8')).hexdigest(),16) % num_modulo
    created_3rd_state = int(hashlib.md5(str(created_2nd_state).encode('utf-8')).hexdigest(),16) % num_modulo
    scale_types, dist_methods, eval_metrics, binary_cols, score_cols = ['t-distribution standardization', 'minmax normalization'], ['manhattan', 'euclidean', 'chebyshev'], ['accuracy', 'error_rate', 'precision', 'recall'], np.array(['gender_num', 'lunch_type', 'has_preparations']), np.array(['math_score', 'reading_score', 'writing_score'])
    
    assignment_params = {}
    assignment_params['num_train'] = 800
    assignment_params['index_col'] = 'student_id'
    assignment_params['o_features'] = ['social_grp', 'parent_edu']
    
    np.random.seed(created_1st_state)
    np.random.seed(123)
    assignment_params['indices_1st'] = list(np.random.choice(12000, 1000, replace=False))
    np.random.seed(created_2nd_state)
    np.random.seed(345)
    assignment_params['indices_2nd'] = list(np.random.permutation(1000))
    
    random.seed(created_3rd_state)
    random.seed(456)
    y_col = random.choice(binary_cols)
    assignment_params['y_col'] = y_col
    assignment_params['binary_cols'] = list(binary_cols[binary_cols!=y_col])
    
    np.random.seed(created_1st_state)
    np.random.seed(789)
    indx_score = sorted(np.random.choice(3, 2, replace=False))
    selected_score_cols = score_cols[indx_score]
    assignment_params['score_col_1'] = selected_score_cols[0]
    assignment_params['score_col_2'] = selected_score_cols[1]
    random.seed(created_1st_state)
    random.seed(789)
    assignment_params['if_correlated_filter'] = random.choice(selected_score_cols)
    
    random.seed(created_2nd_state)  
    assignment_params['scale_type'] = random.choice(scale_types)
    
    random.seed(created_3rd_state)  
    assignment_params['dist_method'] = random.choice(['manhattan', 'euclidean'])
    
    random.seed(created_1st_state)  
    assignment_params['eval_metric'] = random.choice(eval_metrics)
    
    cols_no_filter = assignment_params['o_features'] + assignment_params['binary_cols'] + list(selected_score_cols)
    cols_w_filter = assignment_params['o_features'] + assignment_params['binary_cols'] + [assignment_params['if_correlated_filter']]
    assignment_params['cols_no_filter'] = cols_no_filter
    assignment_params['cols_w_filter'] = cols_w_filter
    
    return assignment_params
# ---------------------------
try:    
    student_name, student_id = myName(), myId()
except Exception as e:
    print ('You probably did not implement student-info functions, \nerror Message:',str(e))
    raise
assert type(student_name) is str or type(student_id) is int, "name is not a string or id is not an integer"         
# ---------------------------
assignment_params = get_assignment_params(student_id)
# ---------------------------
print ('Assignment 1 (- 7 points for the test):')
print ('-----------------------')
print ('What do you need to implement?')
print ('1.   methods: myName, myId - Your personal information (- 0.5 points for the test)')
print ('2.   methods: load_dataset - load the dataset (- 0.5 points for the tests)')
print ("3.   methods: scale_fit_transform, scale_transform_for_test \n\t- the scaling type YOU NEED to implement is: " + assignment_params['scale_type'] + " (- 1.5 points for the tests)")
print ("4.a. methods: calc_distance \n\t- the distance method YOU NEED to implement is: " + assignment_params['dist_method'] + " (- 1.5 points for the tests)")
print ("4.b. methods: predict - KNN predict main flow" + " (- 2 points for the tests)")
print ("5.   methods: evaluate_performance \n\t- the evaluation metric YOU NEED to implement is: " + assignment_params['eval_metric'] + " (- 1 points for the tests)")




# 2.
# ------------>>>>>>>> RUN THIS CODE CELL <<<<<<<<------------
# --------  (run after implementation)
# === CODE TYPE: ANSWER 

def load_dataset(file_name, category_column, index_column):
    df_students = pd.read_csv(file_name)
    return df_students.drop([category_column,index_column],axis=1),df_students
    

# 2.
# ------------>>>>>>>> RUN THIS CODE CELL <<<<<<<<------------
# --------  (run after implementation)
# === CODE TYPE: SELF TESTING
# Use the following code to test your implementation:
file_name = 'data' + os.sep + 'Students-Performance_numeric.csv'
student_name, student_id = myName(), myId()
assignment_params = get_assignment_params(student_id)
X, y = load_dataset(file_name, assignment_params['y_col'], assignment_params['index_col'])
# --- add additional code to check your code if needed:
# YOUR CODE HERE



# 2.
# ------------>>>>>>>> RUN THIS CODE CELL <<<<<<<<------------
# --------  (run after implementation, if used)
# === CODE TYPE: SELF TESTING
# ---- Add your additional tests here if needed:
# YOUR CODE HERE


# 2.
# ------------>>>>>>>> RUN THIS CODE CELL <<<<<<<<------------
# --------  (run only)
# === CODE TYPE: GRADED TEST 

print ("Part 1. - Test 1 (name: test2-1_load_dataset, points: 0.1) - Sanity")
print ("\t--->Testing the implementation of 'load_dataset' ...")

file_name = 'data' + os.sep + 'Students-Performance_numeric.csv'

try:    
    student_name, student_id = myName(), myId()
    assert type(student_name) is str or type(student_id) is int, "name is not a string or id is not an integer"
    assignment_params = get_assignment_params(student_id)
    X, y = load_dataset(file_name, assignment_params['y_col'], assignment_params['index_col'])
except Exception as e:
    print ('You probably have a syntax or implementation error,  \nerror Message:',str(e), '\nTry fixing your code')
    raise

print ("Good Job!\nYou've passed the 1st test for the 'load_dataset' function implementation :-)")

# 2.
# ------------>>>>>>>> RUN THIS CODE CELL <<<<<<<<------------
# --------  (run only)
# === CODE TYPE: GRADED TEST 

print ("Part 1. - Test 2 (name: test2-2_load_dataset, points: 0.1) - Sanity (2)")
print ("\t--->Testing the implementation of 'load_dataset' ...")

file_name = 'data' + os.sep + 'Students-Performance_numeric.csv'

try:    
    student_name, student_id = myName(), myId()
    assert type(student_name) is str or type(student_id) is int, "name is not a string or id is not an integer"
    assignment_params = get_assignment_params(student_id)
    X, y = load_dataset(file_name, assignment_params['y_col'], assignment_params['index_col'])
    # ---------------------------
    X = X[assignment_params['cols_no_filter']]
    X, y = X.iloc[assignment_params['indices_1st'],:], y.iloc[assignment_params['indices_1st']]
    X, y = X.iloc[assignment_params['indices_2nd'],:], y.iloc[assignment_params['indices_2nd']]
    X_train, y_train = X.iloc[:assignment_params['num_train'],:],  y.iloc[:assignment_params['num_train']]
    X_test, y_test = X.iloc[assignment_params['num_train']:,:], y.iloc[assignment_params['num_train']:]
    # ---------------------------
except Exception as e:
    print ('You probably have a syntax or implementation error,  \nerror Message:',str(e), '\nTry fixing your code')
    raise

print ("Good Job!\nYou've passed the 2nd test for the 'load_dataset' function implementation :-)")

# 2.
# ------------>>>>>>>> RUN THIS CODE CELL <<<<<<<<------------
# --------  (run only)
# === CODE TYPE: GRADED TEST 

print ("Part 1. - Test 3 (name: test2-3_load_dataset, points: 0.3)")
print ("\t--->Testing the implementation of 'load_dataset' ...")

file_name = 'data' + os.sep + 'Students-Performance_numeric.csv'

try:    
    student_name, student_id = myName(), myId()
    assert type(student_name) is str or type(student_id) is int, "name is not a string or id is not an integer"
    assignment_params = get_assignment_params(student_id)
    X, y = load_dataset(file_name, assignment_params['y_col'], assignment_params['index_col'])
except Exception as e:
    print ('You probably have a syntax or implementation error,  \nerror Message:',str(e), '\nTry fixing your code')
    raise

assert type(X) is pd.DataFrame or type(y) is pd.Series, "Wrong type for feature vectors dataframe or for category series"
np.testing.assert_array_equal(X.index, y.index, 'X and y should have the same index')
assert X.shape == (12000,7) and X.shape[0] == y.shape[0], 'Wrong shape forfeature vector dataframe or category series'
    
print ("Good Job!\nYou've passed the 3rd test for the 'load_dataset' function implementation :-)")

print()
print()

# 3.a.
# ------------>>>>>>>> RUN THIS CODE CELL <<<<<<<<------------
# --------  (run after implementation)
# === CODE TYPE: ANSWER 

def scale_fit_transform(X_train):
    # YOUR CODE HERE
    copy_X_train = X_train.copy()
    
    cols = []
    means = [0,0,0,0,0]
    stds = [0,0,0,0,0]
    
    data_top = X_train.head()
    for col in data_top.columns:
        cols.append(col)

    for i in range(len(X_train.head())):
        means[i] = np.mean(X_train[cols[i]])

    for i in range(len(X_train.head())):
        for num in X_train[cols[i]]:
            stds[i] += (num-means[i])**2
        stds[i] = np.sqrt(stds[i] * (1/1/len(X_train[cols[i]])))
             
    scale_type = pd.DataFrame(list(zip(means,stds)),
    index=cols,
    columns=['mean', 'std'])

    #print(scale_type)
    print()

    for i in range(len(X_train.head())):
        copy_X_train[cols[i]] = copy_X_train[cols[i]].apply(lambda x: (x-means[i])/stds[i])
    #print(copy_X_train)
    
    
    
    return scale_type,copy_X_train;

# 3.a.
# ------------>>>>>>>> RUN THIS CODE CELL <<<<<<<<------------
# --------  (run after implementation)
# === CODE TYPE: SELF TESTING
# Use the following code to test your implementation:
file_name = 'data' + os.sep + 'Students-Performance_numeric.csv'
student_name, student_id = myName(), myId()
assignment_params = get_assignment_params(student_id)    
X, y = load_dataset(file_name, assignment_params['y_col'], assignment_params['index_col'])
X = X[assignment_params['cols_w_filter']]
X, y = X.iloc[assignment_params['indices_1st'],:], y.iloc[assignment_params['indices_1st']]
X, y = X.iloc[assignment_params['indices_2nd'],:], y.iloc[assignment_params['indices_2nd']]
X_train, y_train = X.iloc[:assignment_params['num_train'],:],  y.iloc[:assignment_params['num_train']]
X_test, y_test = X.iloc[assignment_params['num_train']:,:], y.iloc[assignment_params['num_train']:]
#trained_scaling_info,  X_train_scaled = scale_fit_transform(X_train)
# --- add additional code to check your code if needed:
# YOUR CODE HERE

scale_fit_transform(X_train)


def scale_transform_for_test(trained_scaling_info,  X_test):
    # YOUR CODE HERE
    X_test_scaled = X_test.copy()
    for i in X_test_scaled.columns:
        X_test_scaled[i] = (X_test_scaled[i] - trained_scaling_info.loc[i,'mean'])/trained_scaling_info.loc[i,'std']

    return X_test_scaled
    
# 3.b.
# ------------>>>>>>>> RUN THIS CODE CELL <<<<<<<<------------
# --------  (run after implementation)
# === CODE TYPE: SELF TESTING
# Use the following code to test your implementation:
file_name = 'data' + os.sep + 'Students-Performance_numeric.csv'
student_name, student_id = myName(), myId()
assignment_params = get_assignment_params(student_id)    
X, y = load_dataset(file_name, assignment_params['y_col'], assignment_params['index_col'])
X = X[assignment_params['cols_w_filter']]
X, y = X.iloc[assignment_params['indices_1st'],:], y.iloc[assignment_params['indices_1st']]
X, y = X.iloc[assignment_params['indices_2nd'],:], y.iloc[assignment_params['indices_2nd']]
X_train, y_train = X.iloc[:assignment_params['num_train'],:],  y.iloc[:assignment_params['num_train']]
X_test, y_test = X.iloc[assignment_params['num_train']:,:], y.iloc[assignment_params['num_train']:]
trained_scaling_info,  X_train_scaled = scale_fit_transform(X_train)
X_test_scaled = scale_transform_for_test(trained_scaling_info,  X_test)
# --- add additional code to check your code if needed:
# YOUR CODE HERE




# 4.a.
# ------------>>>>>>>> RUN THIS CODE CELL <<<<<<<<------------
# --------  (run after implementation)
# === CODE TYPE: ANSWER 

def calc_distance(X_test,X_train):
    # YOUR CODE HERE
 
    testArray = []

    print()
    distance = 0
    dist = pd.DataFrame([],
        columns = X_train.index,
        index = X_test.index)
    
    for i in range(len(X_test)):
        calcDistRowTest = X_test.loc[X_test.index[i]]
        for j in range(len(X_train)):
            calcDistRowTrain = X_train.loc[X_train.index[j]]
            distance = euclidean_dist(calcDistRowTrain,calcDistRowTest)
            testArray.append(distance)
        dist.iloc[i] = testArray
        testArray.clear()
        
            
    #print(testArray)
    
    print()    
    print()
    
    #print(dist.loc[1641])
    print()
    #print(X_test.loc[1641])
    print()
    #print(X_train.loc[6731])
            

    #for i in range(20):
        #calcDistRowTest = X_test.loc[X_test.index[i]]
        #for j in range(80):
            #calcDistRowTrain = X_train.loc[X_train.index[j]]
            #distance = np.linalg.norm(calcDistRowTrain-calcDistRowTest)
            #testArray.append(distance)
        #dist.iloc[i] = testArray
        #testArray.clear() 
        
    print()
    print()
    #print(dist)
    
    
    return dist
    

    
    
def euclidean_dist(x,y):
    #3,4,3,1,3
    #4,1,3,4,5
    num=0
    for i in range(len(x)):
        num += (y[i] - x[i])**2
    num = np.sqrt(num)
    return num  
    
    # 4.a.
# ------------>>>>>>>> RUN THIS CODE CELL <<<<<<<<------------
# --------  (run after implementation)
# === CODE TYPE: SELF TESTING
# Use the following code to test your implementation:
file_name = 'data' + os.sep + 'Students-Performance_numeric.csv'
student_name, student_id = myName(), myId()
assignment_params = get_assignment_params(student_id)
X, y = load_dataset(file_name, assignment_params['y_col'], assignment_params['index_col'])
X = X[assignment_params['cols_w_filter']]
X, y = X.iloc[assignment_params['indices_1st'],:], y.iloc[assignment_params['indices_1st']]
X, y = X.iloc[assignment_params['indices_2nd'],:], y.iloc[assignment_params['indices_2nd']]
X_train, y_train = X.iloc[:assignment_params['num_train'],:],  y.iloc[:assignment_params['num_train']]
X_test, y_test = X.iloc[assignment_params['num_train']:,:], y.iloc[assignment_params['num_train']:]
X_train_80, y_train_80 = X_train.iloc[:80,:], y_train.iloc[:80]
X_test_20, y_test_20 = X_test.iloc[:20,:], y_test.iloc[:20]
trained_scaling_info_80,  X_train_scaled_80 = scale_fit_transform(X_train_80)
X_test_scaled_20 = scale_transform_for_test(trained_scaling_info_80,  X_test_20)
dist_dataframe = calc_distance(X_test_20,X_train_80)
dist_df_scaled = calc_distance(X_test_scaled_20,X_train_scaled_80)
# --- add additional code to check your code if needed:
# YOUR CODE HERE


# 4.a.
# ------------>>>>>>>> RUN THIS CODE CELL <<<<<<<<------------
# --------  (run after implementation, if used)
# === CODE TYPE: SELF TESTING
# ---- Add your additional tests here if needed:
# YOUR CODE HERE


# 4.a.
# ------------>>>>>>>> RUN THIS CODE CELL <<<<<<<<------------
# --------  (run only)
# === CODE TYPE: GRADED TEST 

print ("Part 1. - Test 1 (name: test4a-1_calc_distance, points: 0.1) - Sanity")
print ("\t--->Testing the implementation of 'calc_distance' ...")

file_name = 'data' + os.sep + 'Students-Performance_numeric.csv'

try:    
    student_name, student_id = myName(), myId()
    assert type(student_name) is str or type(student_id) is int, "name is not a string or id is not an integer"
    assignment_params = get_assignment_params(student_id)
    print ("\t\tTesing a '" + assignment_params['dist_method'] + "' distance method ...\n")
    X, y = load_dataset(file_name, assignment_params['y_col'], assignment_params['index_col'])
    X = X[assignment_params['cols_w_filter']]
    X, y = X.iloc[assignment_params['indices_1st'],:], y.iloc[assignment_params['indices_1st']]
    X, y = X.iloc[assignment_params['indices_2nd'],:], y.iloc[assignment_params['indices_2nd']]
    X_train, y_train = X.iloc[:assignment_params['num_train'],:],  y.iloc[:assignment_params['num_train']]
    X_test, y_test = X.iloc[assignment_params['num_train']:,:], y.iloc[assignment_params['num_train']:]
    X_train_80, y_train_80 = X_train.iloc[:80,:], y_train.iloc[:80]
    X_test_20, y_test_20 = X_test.iloc[:20,:], y_test.iloc[:20]
    trained_scaling_info_80,  X_train_scaled_80 = scale_fit_transform(X_train_80)
    X_test_scaled_20 = scale_transform_for_test(trained_scaling_info_80,  X_test_20)
    dist_dataframe = calc_distance(X_test_20,X_train_80)
    dist_df_scaled = calc_distance(X_test_scaled_20,X_train_scaled_80)
except Exception as e:
    print ('You probably have a syntax or implementation error,  \nerror Message:',str(e), '\nTry fixing your code')
    raise


print ("Good Job!\nYou've passed the 1st test for the 'calc_distance' function implementation :-)")


# 4.a.
# ------------>>>>>>>> RUN THIS CODE CELL <<<<<<<<------------
# --------  (run only)
# === CODE TYPE: GRADED TEST 

print ("Part 1. - Test 2 (name: test4a-2_calc_distance, points: 0.6)")
print ("\t--->Testing the implementation of 'calc_distance' ...")
file_name = 'data' + os.sep + 'Students-Performance_numeric.csv'

try:    
    student_name, student_id = myName(), myId()
    assert type(student_name) is str or type(student_id) is int, "name is not a string or id is not an integer"
    assignment_params = get_assignment_params(student_id)
    print ("\t\tTesing a '" + assignment_params['dist_method'] + "' distance method ...\n")
    X, y = load_dataset(file_name, assignment_params['y_col'], assignment_params['index_col'])
    X = X[assignment_params['cols_w_filter']]
    X, y = X.iloc[assignment_params['indices_1st'],:], y.iloc[assignment_params['indices_1st']]
    X, y = X.iloc[assignment_params['indices_2nd'],:], y.iloc[assignment_params['indices_2nd']]
    X_train, y_train = X.iloc[:assignment_params['num_train'],:],  y.iloc[:assignment_params['num_train']]
    X_test, y_test = X.iloc[assignment_params['num_train']:,:], y.iloc[assignment_params['num_train']:]
    X_train_80, y_train_80 = X_train.iloc[:80,:], y_train.iloc[:80]
    X_test_20, y_test_20 = X_test.iloc[:20,:], y_test.iloc[:20]
    trained_scaling_info_80,  X_train_scaled_80 = scale_fit_transform(X_train_80)
    X_test_scaled_20 = scale_transform_for_test(trained_scaling_info_80,  X_test_20)
    dist_dataframe = calc_distance(X_test_20,X_train_80)
    dist_df_scaled = calc_distance(X_test_scaled_20,X_train_scaled_80)
    
except Exception as e:
    print ('You probably have a syntax or implementation error,  \nerror Message:',str(e), '\nTry fixing your code')
    raise
assert dist_dataframe.shape == dist_df_scaled.shape, 'Mismatch shape of scaled and non scaled distance dataframes'
np.testing.assert_array_equal(dist_dataframe.index, X_test_scaled_20.index, 'Index of distance dataframe should match test index')
np.testing.assert_array_equal(dist_dataframe.columns, X_train_80.index, 'Column names of distance dataframe should match training index')
np.testing.assert_array_equal(dist_df_scaled.columns, X_train_80.index, 'Column names of distance dataframe should match training index')

print ("Good Job!\nYou've passed the 2nd test for the 'calc_distance' function implementation :-)")



# 4.b.
# ------------>>>>>>>> RUN THIS CODE CELL <<<<<<<<------------
# --------  (run after implementation)
# === CODE TYPE: ANSWER 

def predict(X_test, X_train, y_train, k):
    print(y_train)
    data = calc_distance(X_test,X_train)
    y_predicted = pd.Series(0,index=X_test.index)
    temp = []
    ks = []
    afterKs = []
    arr = []
    
    for i in data.index:
        data.loc[i].sort_values
        t=data.T.sort_values(by=i).T
        temp.append(t.loc[i])

    for i in range(len(X_test)):
        for p in (range(k)):
            ks.append(temp[i].index[p])
        arr.append(ks.copy())
        ks.clear()
        
    #print(arr)
    
    for i in arr:
        afterKs.append(y_train.loc[i,['has_preparations']])
        #afterKs.append(y_train.loc[i])


        
    #print(afterKs)


    
    for i in range(len(X_test)):
        r=0
        for j in range(k):
            r += afterKs[i].values[j] 
        if (r > k/2):
            y_predicted.iloc[i] = 1
        else:
            y_predicted.iloc[i] = 0
            
        
    

    print(y_predicted)
        
    return y_predicted

    

# 4.b.
# ------------>>>>>>>> RUN THIS CODE CELL <<<<<<<<------------
# --------  (run after implementation)
# === CODE TYPE: SELF TESTING
# Use the following code to test your implementation:
file_name = 'data' + os.sep + 'Students-Performance_numeric.csv'
student_name, student_id = myName(), myId()
assignment_params = get_assignment_params(student_id)
X, y = load_dataset(file_name, assignment_params['y_col'], assignment_params['index_col'])
X = X[assignment_params['cols_w_filter']]
X, y = X.iloc[assignment_params['indices_1st'],:], y.iloc[assignment_params['indices_1st']]
X, y = X.iloc[assignment_params['indices_2nd'],:], y.iloc[assignment_params['indices_2nd']]
X_train, y_train = X.iloc[:assignment_params['num_train'],:],  y.iloc[:assignment_params['num_train']]
X_test, y_test = X.iloc[assignment_params['num_train']:,:], y.iloc[assignment_params['num_train']:]
X_train_100, y_train_100 = X_train.iloc[100:200,:], y_train.iloc[100:200]
X_test_20, y_test_20 = X_test.iloc[20:40,:], y_test.iloc[20:40]
trained_scaling_info_100,  X_train_scaled_100 = scale_fit_transform(X_train_100)
X_test_scaled_20 = scale_transform_for_test(trained_scaling_info_100,  X_test_20)
y_predicted = predict(X_test_20, X_train_100, y_train_100, 1)
y_predicted_scaled = predict(X_test_scaled_20, X_train_scaled_100, y_train_100, 1)
# --- add additional code to check your code if needed:
# YOUR CODE HERE








    
    
    





