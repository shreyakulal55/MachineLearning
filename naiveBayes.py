# naive_bayes.py
import matplotlib.pyplot as plt
from sklearn.naive_bayes import GaussianNB
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report
from dataPreprocessing import dataPreprocessing

X_train, X_test, y_train, y_test = dataPreprocessing("C:\\Users\\Shreya\\Downloads\\IPLdata2021.csv")

nb_classifier = GaussianNB()
nb_classifier.fit(X_train, y_train)
y_pred_nb = nb_classifier.predict(X_test)

plt.scatter(y_test, y_pred_nb, color='blue')
plt.xlabel('Actual Values')
plt.ylabel('Predicted Values')
plt.title('Naive Bayes')
plt.show()
print('Naive Bayes Accuracy:', accuracy_score(y_test, y_pred_nb))
print(confusion_matrix(y_test, y_pred_nb))
print(classification_report(y_test, y_pred_nb))
