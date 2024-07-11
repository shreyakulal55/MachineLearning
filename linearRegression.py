import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression
from dataPreprocessing import dataPreprocessing

X_train, X_test, y_train, y_test = dataPreprocessing("C:\\Users\\Shreya\\Downloads\\IPLdata2021.csv")

linear_regressor = LinearRegression()
linear_regressor.fit(X_train, y_train)
y_pred_linear = linear_regressor.predict(X_test)

plt.scatter(y_test, y_pred_linear, color='blue')
plt.xlabel('Actual Values')
plt.ylabel('Predicted Values')
plt.title('Linear Regression')
plt.show()
print('Linear Regression R^2 Score:', linear_regressor.score(X_test, y_test))
