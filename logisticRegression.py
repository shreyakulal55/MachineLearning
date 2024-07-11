import matplotlib.pyplot as plt
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report
from dataPreprocessing import dataPreprocessing

X_train, X_test, y_train, y_test = dataPreprocessing("C:\\Users\\Shreya\\Downloads\\IPLdata2021.csv")

logistic_regressor = LogisticRegression(max_iter=1000)
logistic_regressor.fit(X_train, y_train)
y_pred_logistic = logistic_regressor.predict(X_test)

plt.scatter(y_test, y_pred_logistic, color='blue')
plt.xlabel('Actual Values')
plt.ylabel('Predicted Values')
plt.title('Logistic Regression')
plt.show()
print('Logistic Regression Accuracy:', accuracy_score(y_test, y_pred_logistic))
print(confusion_matrix(y_test, y_pred_logistic))
print(classification_report(y_test, y_pred_logistic))
