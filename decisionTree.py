import matplotlib.pyplot as plt
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report
from dataPreprocessing import dataPreprocessing

X_train, X_test, y_train, y_test = dataPreprocessing("C:\\Users\\Shreya\\Downloads\\IPLdata2021.csv")

tree_classifier = DecisionTreeClassifier()
tree_classifier.fit(X_train, y_train)
y_pred_tree = tree_classifier.predict(X_test)

plt.scatter(y_test, y_pred_tree, color='blue')
plt.xlabel('Actual Values')
plt.ylabel('Predicted Values')
plt.title('Decision Tree')
plt.show()
print('Decision Tree Accuracy:', accuracy_score(y_test, y_pred_tree))
print(confusion_matrix(y_test, y_pred_tree))
print(classification_report(y_test, y_pred_tree))
