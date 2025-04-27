#just a quick hypothetical example of a stroke prediction model using the stroke prediction dataset from Kaggle.

#future ref: kaggle competitions download -c [COMPETITION]: download files associated with a competition

import kagglehub # Download dataset from Kaggle
import pandas as pd # Data manipulation and analysis library
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.impute import SimpleImputer
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score, confusion_matrix, classification_report


# Download latest version
path = kagglehub.dataset_download("fedesoriano/stroke-prediction-dataset")

print("Path to dataset files:", path)



# loads CSV file into a pandas DataFrame afterward
df = pd.read_csv(path + "/healthcare-dataset-stroke-data.csv")  
print(df.head()) #prints DS preview
print(df.info()) #prints DS info

#quick exploratory data analysis (EDA)

print("\n missing values per column")
print(df.isnull().sum()) #prints missing values per column

print("\n target var distribution")
print(df['stroke'].value_counts(normalize=True)) #prints target var distribution

sns.countplot(data=df,x='stroke') 
plt.title('Cumulative stroke distribution')
plt.show()

#data cleaning
#handling of missing values in bmi
imputer=SimpleImputer(strategy='mean')  #mean imputation
df['bmi']=imputer.fit_transform(data[['bmi']]) #imputes missing values in bmi column

#dropp id in column 
df.drop('id', axis=1, inplace=True) #drops id column

#encoding of categorical vauoiables
cat_cols=df.select_dtypes(include='object').columns  
encoder=LabelEncoder()
for col in cat_cols:
    df[col]=encoder.fit_transform(df[col]) #encodes categorical variables


#model building
X = df.drop('stroke', axis=1) #features
Y = df['stroke'] #target variable

X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.2, random_state=42, stratify=Y) #splitting of dataset into train/test subsets

#logistic regression  
log_reg=LogisticRegression()    #fits logistic regression model
log_reg.fit(X_train, Y_train)    #predicts on test set
y_pred_logreg=log_reg.predict(X_test)   #predicts on test set

#random forest classifier
rf=RandomForestClassifier(n_estimators=1000)  #fits random forest classifier
rf.fit(X_train, Y_train)    #prediction on test set
y_pred_rf=rf.predict(X_test)  #predicts on test set

#model evaliuation
def evaluate_model(y_true, y_pred, model_name):
    print(f"\n{model_name} Model Evaluation:")
    print(f"Accuracy: {accuracy_score(y_true, y_pred):.4f}")
    print(f"Precision: {precision_score(y_true, y_pred):.4f}")
    print(f"Recall: {recall_score(y_true, y_pred):.4f}")
    print(f"F1 Score: {f1_score(y_true, y_pred):.4f}")
    print(f"ROC AUC Score: {roc_auc_score(y_true, y_pred):.4f}")
    print("\nClassification Report:")
    print(classification_report(y_true, y_pred))
    
    #confusion matrix
    print("\nConfusion Matrix:")
    cm = confusion_matrix(y_true, y_pred)
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues')
    plt.xlabel('Predicted')
    plt.ylabel('True')
    plt.title(f'{model_name} Confusion Matrix')
    plt.show()

#evaallujation on both models;
evaluate_model(Y_test, y_pred_logreg, "Logistic Regression")
evaluate_model(Y_test, y_pred_rf, "Random Forest")

#feature importance: RF model
importances=rf.feature_importances_  #feature importance
feat_importances=pd.Series(importances, index=X.columns)  #creates a pandas series
feat_importances.nlargest(10).plot(kind='barh') #plots top 10 features
plt.title('Feature Importance - Random Forest')
plt.show()

#running output: is an imbalanced classification problem

#NO: 0.95 (0)
#YES: 0.05 (1)

#FOCUS ON RECALL AND F1 SCORE VALUATIONS, NOT JSUT ACCURACY!