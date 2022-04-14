# import statements 
import pickle
import numpy as np
import pandas as pd
from sklearn import metrics
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from sklearn.metrics import confusion_matrix
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import cross_val_score
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier


#spilt the dataset and save into different files
df=pd.read_csv("heart.csv")
X = df.drop(columns=['HeartDisease'])
y = df.HeartDisease
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, stratify=y)

train=pd.DataFrame(X_train, y_train)
train.to_csv('Heart_train_data.csv')

test=pd.DataFrame(X_test, y_test)
test.to_csv('Heart_test_data.csv')

# Filename of the dataset to use for training and validation
train_data = "Heart_train_data.csv"
# Filename of test dataset to apply your model and predict outcomes 
test_data = "Heart_test_data.csv"



# Load the trainig data, clean/prepare and obtain training and target vectors, 
def load_prepare():
    train_data=pd.read_csv("heart.csv")
    train_data['Sex'] = train_data['Sex'].apply(lambda x: 0 if x=='F' else 1 if x=='M' else np.nan)
    train_data['ExerciseAngina'] = train_data['ExerciseAngina'].apply(lambda x: 0 if x=='N' else 1 if x=='Y' else np.nan)
    train_data=pd.get_dummies(train_data,columns=['ChestPainType'])
    train_data=pd.get_dummies(train_data,columns=['RestingECG'])
    train_data=pd.get_dummies(train_data,columns=['ST_Slope'])
    # return training vector and target vector
    X = train_data.drop(columns=['HeartDisease'])
    y = train_data.HeartDisease
    return X, y

# Split it into train/validate sets
# Build a pipeline to transform the training vector and fit an appropriate machine learning model
# Validate your model accuracy using the validation set
def build_pipeline_1(X, y):
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, stratify=y)
    pipeline=Pipeline([
        ('std_scaler',StandardScaler()),
        ('lr_model', LogisticRegression(penalty='none', max_iter=1000))
    ])
    pipeline.fit(X_train, y_train)
    y_predict = pipeline.predict(X_test)

    #training_accuracy = accuracy_score(y_test, y_predict).round(4)
    scores = cross_val_score(pipeline, X_train, y_train, cv=5)
    training_accuracy = np.mean(scores).round(4)
    confusion_matrix= metrics.confusion_matrix(y_test, y_predict)
    pickle.dump(pipeline, open('pipeline.pkl', 'wb'))
    # return training accuracy, sklearn confusion matrix (from validation step) and sklearn pipeline object
    return training_accuracy, confusion_matrix, pipeline


# Split it into train/validate sets
# Build a pipeline to transform the training vector and fit an appropriate machine learning model
# Validate your model accuracy using the validation set
def build_pipeline_2(X, y):
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, stratify=y)

    pipeline=Pipeline([
        ('std_scaler',StandardScaler()),
        ('knn_model', KNeighborsClassifier(n_neighbors=10))
    ])
    pipeline.fit(X_train, y_train)
    y_predict = pipeline.predict(X_test)

    scores = cross_val_score(pipeline, X_train, y_train, cv=5)
    training_accuracy = np.mean(scores).round(4)
    confusion_matrix= metrics.confusion_matrix(y_test, y_predict)
    # return training accuracy, sklearn confusion matrix (from validation step) and sklearn pipeline object
    return training_accuracy, confusion_matrix, pipeline


# This your final and improved model pipeline
# Split it into train/validate sets
# Build a pipeline to transform the training vector and fit an appropriate machine learning model
# Validate your model accuracy using the validation set
# Save your final pipeline to a file "pipeline.pkl"   
def build_pipeline_final(X, y):
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, stratify=y)

    pipeline=Pipeline([
        ('std_scaler',StandardScaler()),
        ('dt_model', DecisionTreeClassifier(max_depth=6))
    ])
    pipeline.fit(X_train, y_train)
    y_predict = pipeline.predict(X_test)

    scores = cross_val_score(pipeline, X_train, y_train, cv=5)
    training_accuracy = np.mean(scores).round(4)
    confusion_matrix= metrics.confusion_matrix(y_test, y_predict)
    #pickle.dump(pipeline, open('pipeline.pkl', 'wb'))
    # return training accuracy, sklearn confusion matrix (from validation step) and sklearn pipeline object
    return training_accuracy, confusion_matrix, pipeline


# Load final pipeline (pipe.pkl) and test dataset (test_data)
# Apply the pipeline to the test data and predict outcomes
def apply_pipeline():
    pipeline = pickle.load(open('pipeline.pkl','rb'))
    X, y = load_prepare()
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, stratify=y)
    pipeline.fit(X_train,y_train)
    predictions =pipeline.predict(X_test)

    # return predictions or outcomes
    return predictions

