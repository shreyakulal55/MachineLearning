# random_forest.py
import matplotlib.pyplot as plt
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report
from dataPreprocessing import dataPreprocessing

X_train, X_test, y_train, y_test = dataPreprocessing("C:\\Users\\Shreya\\Downloads\\IPLdata2021.csv")

forest_classifier = RandomForestClassifier(n_estimators=100)
forest_classifier.fit(X_train, y_train)
y_pred_forest = forest_classifier.predict(X_test)

plt.scatter(y_test, y_pred_forest, color='blue')
plt.xlabel('Actual Values')
plt.ylabel('Predicted Values')
plt.title('Random Forest')
plt.show()
print('Random Forest Accuracy:', accuracy_score(y_test, y_pred_forest))
print(confusion_matrix(y_test, y_pred_forest))
print(classification_report(y_test, y_pred_forest))
