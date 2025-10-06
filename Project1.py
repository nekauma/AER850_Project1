#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Sep 30 16:13:28 2025

@author: nekaumakanth
"""

# 2.1 data processing, read data from csv file and convert into dataframe 
import pandas as pd
data = pd.read_csv("Project1Data.csv")
df = pd.DataFrame(data)

# 2.2 Data Visualization 
import numpy as np
import matplotlib.pyplot as plt
from matplotlib import cm 
# Extract columns 
X_coord = df['X']
Y_coord = df['Y']
Z_coord = df['Z']
Steps = df['Step']
u_steps = df['Step'].unique() # distinct step labels for each step, to loop through 
summary_list = [] #store statistical data 

cmap = cm.get_cmap('jet', len(u_steps))


for step in u_steps:
    # loops through each step 
    step_rows = df[df['Step'] == step]
    
    # extract X, Y, Z as NumPy arrays for easier numeric calculation
    x = step_rows['X'].values
    y = step_rows['Y'].values
    z = step_rows['Z'].values
    
    # compute stats
    count = len(x)
    mean = [np.mean(x), np.mean(y), np.mean(z)]
    std = [np.std(x), np.std(y), np.std(z)] #standard deviation
    min_val = [np.min(x), np.min(y), np.min(z)] #minimum values
    max_val = [np.max(x), np.max(y), np.max(z)] #maximum values 
    
    summary_list.append([step, count, *mean, *std, *min_val, *max_val])
columns = ['Step', 'Count',
           'Mean_X','Mean_Y','Mean_Z',
           'Std_X','Std_Y','Std_Z',
           'Min_X','Min_Y','Min_Z',
           'Max_X','Max_Y','Max_Z']

summary_df = pd.DataFrame(summary_list, columns=columns) # list to dataframe
summary_df = summary_df.round(2)
print(summary_df)
summary_df.to_csv("Stats_Summary.csv", index=False)

# Want to  plot x vs y, x vs z, y vs z with respective step 
# Plot 1 - X vs Y 
# Loops through each unique step, i = index, step = actual step number to filter rows
plt.figure()
for i, step in enumerate(u_steps):
    step_rows = df[df['Step']==step]
    plt.scatter(
        step_rows['X'],
        step_rows['Y'],
        label=f'Step {step}',
        color=cmap(i)
    )
plt.xlabel('X_coord')
plt.ylabel('Y_coord')
plt.title('X vs Y Coordinates Coloured by Step')
plt.legend(bbox_to_anchor=(1, 1), loc='upper left')
plt.show()


# Plot 2 - X vs Z
plt.figure(2)
for i, step in enumerate(u_steps):
    step_rows = df[df['Step']==step]
    plt.scatter(
        step_rows['X'],
        step_rows['Z'],
        label=f'Step {step}',
        color=cmap(i)
    )
plt.xlabel('X_coord')
plt.ylabel('Z_coord')
plt.title('X vs Z Coordinates Coloured by Step')
plt.legend(bbox_to_anchor=(1, 1), loc='upper left')
plt.show()
# Plot 3 - Y vs Z
plt.figure(3)
for i, step in enumerate(u_steps):
    step_rows = df[df['Step']==step]
    plt.scatter(
        step_rows['Y'],
        step_rows['Z'],
        label=f'Step {step}',
        color=cmap(i)
    )
plt.xlabel('Y_coord')
plt.ylabel('Z_coord')
plt.title('Y vs Z Coordinates Coloured by Step')
plt.legend(bbox_to_anchor=(1, 1), loc='upper left')
plt.show()

# Plot 4 - 3D Scatter plot for easier visualization 
fig = plt.figure()
# 3D axes 
ax = fig.add_subplot(111, projection='3d')
for i, step in enumerate(u_steps):
    step_rows = df[df['Step']==step]
    ax.scatter(
        step_rows['X'],
        step_rows['Y'],
        step_rows['Z'],
        label=f'Step {step}',
        color=cmap(i)
    )
ax.set_xlabel('X_coord')
ax.set_ylabel('Y_coord')
ax.set_zlabel('Z_coord')
ax.set_title('XYZ Coordinates Coloured by Step')
ax.legend(bbox_to_anchor=(1.3, 1), loc='upper left')
plt.show()

# Step 3: Correlation Analysis
import seaborn as sns
corr_matrix = data.corr()
print(corr_matrix)
sns.heatmap((corr_matrix)) 

# Step 4: Classification Model Development
# Splitting data into 80% train 20% test
coords = df[['X','Y','Z']]
from sklearn.model_selection import train_test_split
X_train, X_test, Y_train, Y_test = train_test_split(coords, Steps,
                                                    test_size= 0.2,
                                                    random_state = 42,
                                                    stratify = Steps
                                                    )
# Ensuring data is split proportionality (Stratified)
print("Train set class counts:\n", Y_train.value_counts())
print("Test set class counts:\n", Y_test.value_counts())

# Model 1 - Random Forest based on RandomizedSearchCV
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import RandomizedSearchCV
# hyperparameters
rf_param_grid = {
    'n_estimators': [50, 100, 150],
    'max_depth': [5, 10, 15],
    'min_samples_split': [2, 3, 5],
    'min_samples_leaf': [1, 2, 3],
    'max_features': ['sqrt', 'log2'],
}
# cross-validation with Randomized Search 
rf = RandomForestClassifier(random_state=42)
rf_rnd_search = RandomizedSearchCV(
    estimator= rf,
    n_iter=20,
    param_distributions=rf_param_grid,
    cv=5,
    scoring='f1_weighted',
    n_jobs=-1
)
# fitting to train data
rf_rnd_search.fit(X_train, Y_train)
# best parameters from param grid
print(rf_rnd_search.best_params_) # best params

from sklearn.model_selection import GridSearchCV
from sklearn.neighbors import KNeighborsClassifier
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
# Model 2 - K-Nearest Neighbour
# Data Scaling - pipeline
pipeline1 = Pipeline([
    ('scaler', StandardScaler()),
    ('model', KNeighborsClassifier(n_jobs=-1))
    ])


knn_param_grid = {
    'model__n_neighbors': [3, 5, 7],
    'model__weights': ['uniform','distance'],
    'model__metric':['euclidean','manhattan']
    }
# cross-validation with Grid Search 
knn_grid_search = GridSearchCV(
    estimator=pipeline1,
    param_grid=knn_param_grid,
    cv = 5,
    scoring = 'f1_weighted',
    n_jobs=-1
    )

knn_grid_search.fit(X_train, Y_train)
print(knn_grid_search.best_params_)
# Model 3 - SVM
from sklearn.svm import SVC 
pipeline2 = Pipeline([
    ('scaler', StandardScaler()),
    ('model',SVC(random_state=42))
    ])
svm_param_grid = {
    'model__C': [0.01, 0.1, 1, 10],
    'model__kernel':['linear','rbf','poly'],
    'model__gamma':['scale','auto']
    }
svm_grid_search = GridSearchCV(
    estimator=pipeline2,
    param_grid=svm_param_grid,
    cv = 5,
    scoring = 'f1_weighted',
    n_jobs=-1
    )

svm_grid_search.fit(X_train, Y_train)
print(svm_grid_search.best_params_)

# Step 5: Model Performance Analysis
# RF model evaluation
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
best_rfparams = rf_rnd_search.best_params_
rf_best = RandomForestClassifier(**best_rfparams, random_state=42) # selects best params
rf_best.fit(X_train, Y_train) 
y_pred_rf = rf_best.predict(X_test) # Predict steps for test set

# Performance metrics
acc_rf = accuracy_score(Y_test, y_pred_rf)
prec_rf = precision_score(Y_test, y_pred_rf, average='weighted')
rec_rf = recall_score(Y_test, y_pred_rf, average='weighted')
f1_rf = f1_score(Y_test, y_pred_rf, average='weighted')

# K-Nearest Neighbour model
best_knn = knn_grid_search.best_estimator_ # .estimator retrives pipeline with best params
y_pred_knn = best_knn.predict(X_test)

acc_knn = accuracy_score(Y_test, y_pred_knn)
prec_knn = precision_score(Y_test, y_pred_knn, average='weighted')
rec_knn = recall_score(Y_test, y_pred_knn, average='weighted')
f1_knn = f1_score(Y_test, y_pred_knn, average='weighted')


# SVM Model
best_svm = svm_grid_search.best_estimator_
y_pred_svm = best_svm.predict(X_test)

acc_svm = accuracy_score(Y_test, y_pred_svm)
prec_svm = precision_score(Y_test, y_pred_svm, average='weighted')
rec_svm = recall_score(Y_test, y_pred_svm, average='weighted')
f1_svm = f1_score(Y_test, y_pred_svm, average='weighted')

# Metrics into DataFrame for comparison 
metrics_dict = {
    'Accuracy': [acc_rf, acc_knn, acc_svm],
    'Precision': [prec_rf, prec_knn, prec_svm],
    'Recall': [rec_rf, rec_knn, rec_svm],
    'F1-score': [f1_rf, f1_knn, f1_svm]
}
metrics_df = pd.DataFrame(metrics_dict, index=['Random Forest', 'KNN', 'SVM'])
print(metrics_df)

from sklearn.metrics import confusion_matrix

# Choose the model with highest F1-score (SVM in this case)
best_model_name = 'SVM'
y_pred_best = y_pred_svm 

# Create confusion matrix
cm_svm = confusion_matrix(Y_test, y_pred_svm, labels=np.unique(Steps))
fig, ax = plt.subplots(figsize=(8, 6))
sns.heatmap(cm_svm, annot=True, fmt='d', cmap='RdPu',
            xticklabels=np.unique(Steps),
            yticklabels=np.unique(Steps),
            ax=ax)

plt.title(f'Confusion Matrix for {best_model_name}')
plt.xlabel("Predicted")
plt.ylabel("Actual")
plt.show()

# Step 6: Stacked Model Peformance
from sklearn.ensemble import StackingClassifier
from sklearn.linear_model import LogisticRegression
# RF + KNN
stacked_model = StackingClassifier(
    estimators=[
        ('rf', rf_best), #combining best params from both
        ('knn', best_knn)
    ],
    final_estimator=LogisticRegression(max_iter=1000), #learn from base models
    cv=5,
    n_jobs=-1
)
stacked_model.fit(X_train, Y_train)
y_pred_stack = stacked_model.predict(X_test) # predicted test labels for stacked model

# Performance metrics computed
acc_stack = accuracy_score(Y_test, y_pred_stack)
prec_stack = precision_score(Y_test, y_pred_stack, average='weighted')
rec_stack = recall_score(Y_test, y_pred_stack, average='weighted')
f1_stack = f1_score(Y_test, y_pred_stack, average='weighted')

# confusion matrix for stacked model
cm_stack = confusion_matrix(Y_test, y_pred_stack)
plt.figure(figsize=(8,6))
sns.heatmap(cm_stack, annot=True, fmt='d', cmap='RdPu')
plt.title("Stacked Model Confusion Matrix (RF + KNN)")
plt.xlabel("Predicted")
plt.ylabel("Actual")
plt.show()

# Table Form
metrics_dict['Accuracy'].append(acc_stack)
metrics_dict['Precision'].append(prec_stack)
metrics_dict['Recall'].append(rec_stack)
metrics_dict['F1-score'].append(f1_stack)

metrics_df = pd.DataFrame(metrics_dict, index=['Random Forest', 'KNN', 'SVM', 'RF+KNN Stacked'])
print(metrics_df)

# Step 7: Model Evaluation 
import joblib
joblib.dump(best_svm, 'svm_model.joblib')
loaded_model = joblib.load('svm_model.joblib')

new_coords = np.array([
    [9.375, 3.0625, 1.51],
    [6.995, 5.125, 0.3875],
    [0, 3.0625, 1.93],
    [9.4, 3, 1.8],
    [9.4, 3, 1.3]
])

predictions = loaded_model.predict(new_coords)
print("Predicted steps for random coordinates:", predictions)




