import matplotlib.pyplot as plt
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report, hamming_loss
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.model_selection import train_test_split
import pandas as pd

def preprocess_data(file_path):
    df = pd.read_csv(file_path)

    df = df.drop(columns=['Unnamed: 0', 'Name'])

    X = df.drop(columns=['fours', 'sixes'])
    y_fours = df['fours'] 
    y_sixes = df['sixes'] 
    label_encoder = LabelEncoder()
    y_fours = label_encoder.fit_transform(y_fours)
    y_sixes = label_encoder.fit_transform(y_sixes)

    scaler = StandardScaler()
    X = scaler.fit_transform(X)

    X_train, X_test, y_train_fours, y_test_fours = train_test_split(X, y_fours, test_size=0.2, random_state=42)
    _, _, y_train_sixes, y_test_sixes = train_test_split(X, y_sixes, test_size=0.2, random_state=42)

    return X_train, X_test, y_train_fours, y_test_fours, y_train_sixes, y_test_sixes

X_train, X_test, y_train_fours, y_test_fours, y_train_sixes, y_test_sixes = preprocess_data("C:\\Users\\Shreya\\Downloads\\IPLdata2021.csv")

svm_classifier_fours = SVC()
svm_classifier_fours.fit(X_train, y_train_fours)

y_pred_fours = svm_classifier_fours.predict(X_test)

svm_classifier_sixes = SVC()
svm_classifier_sixes.fit(X_train, y_train_sixes)

y_pred_sixes = svm_classifier_sixes.predict(X_test)

plt.scatter(y_test_fours, y_pred_fours, color='blue', label='Fours')
plt.scatter(y_test_sixes, y_pred_sixes, color='red', label='Sixes')

plt.xlabel('Actual Values')
plt.ylabel('Predicted Values')
plt.title('Actual vs Predicted Fours and Sixes')
plt.legend()
plt.show()

print('Fours - Accuracy:', accuracy_score(y_test_fours, y_pred_fours))
print('Fours - Classification Report:')
print(classification_report(y_test_fours, y_pred_fours))

print('Sixes - Accuracy:', accuracy_score(y_test_sixes, y_pred_sixes))
print('Sixes - Classification Report:')
print(classification_report(y_test_sixes, y_pred_sixes))
