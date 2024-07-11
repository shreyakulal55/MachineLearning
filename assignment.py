# Importing necessary libraries
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LinearRegression, LogisticRegression
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score
import matplotlib.pyplot as plt
import seaborn as sns

# Load the dataset
data = pd.read_csv("C:\\Users\\Shreya\\Downloads\\IPLdata2021.csv")

# Display the first few rows of the dataset
print(data.head())

# Drop the 'Unnamed: 0' column as it is not needed
data = data.drop(columns=['Unnamed: 0'])

# Convert the 'Name' column to numerical values using one-hot encoding
data = pd.get_dummies(data, columns=['Name'], drop_first=True)

# Display the first few rows of the dataset after encoding
print(data.head())

# Split the dataset into features and target variable
X = data.drop(columns=['sr']).values
y = data['sr'].values

# Split the dataset into training and test sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=0)

# Feature scaling
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

# Linear Regression
linear_regressor = LinearRegression()
linear_regressor.fit(X_train, y_train)
y_pred_linear = linear_regressor.predict(X_test)
print('Linear Regression R^2 Score:', linear_regressor.score(X_test, y_test))

# Logistic Regression
logistic_regressor = LogisticRegression()
logistic_regressor.fit(X_train, y_train)
y_pred_logistic = logistic_regressor.predict(X_test)
print('Logistic Regression Accuracy:', accuracy_score(y_test, y_pred_logistic))
print(confusion_matrix(y_test, y_pred_logistic))
print(classification_report(y_test, y_pred_logistic))

# Support Vector Machine (SVM)
svm_classifier = SVC()
svm_classifier.fit(X_train, y_train)
y_pred_svm = svm_classifier.predict(X_test)
print('SVM Accuracy:', accuracy_score(y_test, y_pred_svm))
print(confusion_matrix(y_test, y_pred_svm))
print(classification_report(y_test, y_pred_svm))

# Decision Tree
tree_classifier = DecisionTreeClassifier()
tree_classifier.fit(X_train, y_train)
y_pred_tree = tree_classifier.predict(X_test)
print('Decision Tree Accuracy:', accuracy_score(y_test, y_pred_tree))
print(confusion_matrix(y_test, y_pred_tree))
print(classification_report(y_test, y_pred_tree))

# Random Forest
forest_classifier = RandomForestClassifier(n_estimators=100)
forest_classifier.fit(X_train, y_train)
y_pred_forest = forest_classifier.predict(X_test)
print('Random Forest Accuracy:', accuracy_score(y_test, y_pred_forest))
print(confusion_matrix(y_test, y_pred_forest))
print(classification_report(y_test, y_pred_forest))

# Naive Bayes
nb_classifier = GaussianNB()
nb_classifier.fit(X_train, y_train)
y_pred_nb = nb_classifier.predict(X_test)
print('Naive Bayes Accuracy:', accuracy_score(y_test, y_pred_nb))
print(confusion_matrix(y_test, y_pred_nb))
print(classification_report(y_test, y_pred_nb))

# K-Nearest Neighbors (KNN)
knn_classifier = KNeighborsClassifier()
knn_classifier.fit(X_train, y_train)
y_pred_knn = knn_classifier.predict(X_test)
print('KNN Accuracy:', accuracy_score(y_test, y_pred_knn))
print(confusion_matrix(y_test, y_pred_knn))
print(classification_report(y_test, y_pred_knn))

# Data Visualization
# Plotting the correlation matrix
plt.figure(figsize=(10, 8))
sns.heatmap(data.corr(), annot=True, cmap='coolwarm')
plt.title('Correlation Matrix')
plt.show()

# Pairplot
sns.pairplot(data)
plt.show()

# Feature importance from Random Forest
feature_importance = forest_classifier.feature_importances_
features = data.columns[:-1]
plt.figure(figsize=(10, 8))
sns.barplot(x=feature_importance, y=features)
plt.title('Feature Importance from Random Forest')
plt.show()
