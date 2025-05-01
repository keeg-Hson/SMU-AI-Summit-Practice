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
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_curve, roc_auc_score, confusion_matrix, classification_report


# loads CSV file (dataset) into a pandas DataFrame afterward
url = "https://raw.githubusercontent.com/jbrownlee/Datasets/master/pima-indians-diabetes.data.csv"
columns = ['Pregnancies', 'Glucose', 'BloodPressure', 'SkinThickness', 'Insulin', 'BMI', 'DiabetesPedigreeFunction', 'Age', 'Outcome']
df = pd.read_csv(url, names=columns)  


#previewing
print(df.head()) #prints DS preview
print(df.info()) #prints DS info
print(df['Outcome'].value_counts(normalize=True)) #prints DS value counts

#dealss with zeroes, replaces zero indicies with NaN
cols_with_zero=['BloodPressure', 'Glucose', 'BMI']
df[cols_with_zero]=df[cols_with_zero].replace(0,np.nan)

#imputes missing values with mean
imputer=SimpleImputer(strategy='mean')
df[cols_with_zero]=imputer.fit_transform(df[cols_with_zero])

#feat/target splitting
X=df.drop('Outcome', axis=1)
Y=df['Outcome']

#train/test splitting
X_train, X_test, Y_train, Y_test = train_test_split(X,Y,test_size=0.2, random_state=42, stratify=Y)

#model training
log_reg=LogisticRegression(max_iter=1000, class_weight='balanced')
log_reg.fit(X_train,Y_train)
y_pred_logreg=log_reg.predict(X_test)
y_prob_logreg=log_reg.predict_proba(X_test)[:,1]

rf = RandomForestClassifier(n_estimators=200, random_state=42)
rf.fit(X_train, Y_train)
y_pred_rf = rf.predict(X_test)
y_prob_rf=rf.predict_proba(X_test)[:,1]

#functional evaluation
def evaluate(y_true, y_pred, y_proba, name):
    print(f"Evaluation for {name} Results:")
    print("Accuracy:", accuracy_score(y_true, y_pred))
    print("Precision:", precision_score(y_true, y_pred))
    print("Recall:", recall_score(y_true, y_pred))
    print("F1 Score:", f1_score(y_true, y_pred))
    print("ROC AUC Score:", roc_auc_score(y_true, y_proba))
    print("\nConfusion Matrix:\n", confusion_matrix(y_true, y_pred))
    print("\nClassification Report:\n", classification_report(y_true, y_pred))


    #CONFUSION MATRIX
    sns.heatmap(confusion_matrix(y_true, y_pred), annot=True, fmt='d', cmap='Blues')
    plt.title(f"{name} Confusion matrix")
    plt.xlabel('predicted')
    plt.ylabel('true')
    plt.show()

    #ROC curve
    fpr, tpr, _ = roc_curve(y_true, y_proba)
    plt.plot(fpr, tpr, label=f'{name} (AUC = {roc_auc_score(y_true, y_proba):.2f})')
    plt.plot([0,1],[0,1],'k--')
    plt.xlabel('false positive rate')
    plt.ylabel('true positive rate')
    plt.title(f'{name} ROC curve')
    plt.legend()
    plt.show()

#running evaluations
evaluate(Y_test, y_pred_logreg, y_prob_logreg, "Logistic Regression")
evaluate(Y_test, y_pred_rf, y_prob_rf, "Random Forest")
    

