import streamlit as st 
import pandas as pd 
import numpy as np
import seaborn as sns
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC 
from sklearn.ensemble import RandomForestClassifier
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix, accuracy_score, precision_score, recall_score, f1_score
import matplotlib.pyplot as plt 


st.title("Billboard Hot 100 Analysis and Machine Learning in Python from (1958-2020)")

st.write("""
## Explore different classifiers on the 
## Billboard Hot 100 from 1958 to 2020
## which one is best for predicting the No. 1 Spot? 
         """)

dataset_name = st.sidebar.selectbox("Select Dataset", ("ML_DF", "ML_DF"))

classifier_name = st.sidebar.selectbox("Select Classifier", ("KNN", "SVM", "Random Forest"))

def get_dataset(dataset_name):
    if dataset_name == "ML_DF":
        data = pd.read_pickle('ml_df.pkl')
    else:
        data = pd.read_pickle('ml_df.pkl')
    X = data.drop(columns='top_hit_pos')
    y = data['top_hit_pos']
    numerical_ix = X.select_dtypes(include=['int64', 'float64']).columns
    categorical_ix = X.select_dtypes(include=['object', 'bool']).columns
    t = [('cat', OneHotEncoder(), categorical_ix), ('num', MinMaxScaler(), numerical_ix)]
    col_transform = ColumnTransformer(transformers=t)
    X = col_transform.fit_transform(X)
    return X, y 

X, y = get_dataset(dataset_name)
st.write("Shape of Features", X.shape)
st.write("Number of classes", len(np.unique(y)))

def add_parameter_ui(clf_name):
    params = dict()
    if clf_name == "KNN":
        K = st.sidebar.slider("K", 1, 15)
        params["K"] = K
    elif clf_name == "SVM":
        C = st.sidebar.slider("C", 0.1, 10.0)
        params["C"] = C
    else:
        max_depth = st.sidebar.slider("max_depth", 2, 15)
        n_estimators = st.sidebar.slider("n_estimators", 1, 100)
        min_samples_split = st.sidebar.slider("min_samples_split", 2, 20)
        params["min_samples_split"] = min_samples_split
        params["max_depth"] = max_depth
        params["n_estimators"] = n_estimators
    return params

params = add_parameter_ui(classifier_name)

def get_classifier(clf_name, params):
    if clf_name == "KNN":
        clf = KNeighborsClassifier(n_neighbors=params["K"])
    elif clf_name == "SVM":
        clf = SVC(C=params["C"])
    else:
        clf = RandomForestClassifier(n_estimators=params["n_estimators"],
                                     max_depth=params["max_depth"], 
                                     min_samples_split=params["min_samples_split"], random_state=42)
    return clf

clf = get_classifier(classifier_name, params)

# Classificatoin
# Split data into training and test sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

clf.fit(X_train, y_train)
y_pred = clf.predict(X_test)

acc = accuracy_score(y_test, y_pred)
st.write(f"classifier = {classifier_name}")
st.write(f"accuracy = {acc}")

# Plot
# Create confusion matrix object
cm = confusion_matrix(y_test, y_pred)

# Create a heatmap plot of the confusion matrix
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues')
plt.xlabel('Predicted label')
plt.ylabel('True label')

# plt.show()
st.set_option('deprecation.showPyplotGlobalUse', False)
st.pyplot()
