import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

def dataPreprocessing(file_path):
    data = pd.read_csv(file_path)

    print("Original Data is:\n", data.head())

    if 'Unnamed: 0' in data.columns:
        data = data.drop(columns=['Unnamed: 0'])

    numeric_columns = data.select_dtypes(include=['number']).columns
    data[numeric_columns] = data[numeric_columns].fillna(data[numeric_columns].mean())
    
    non_numeric_columns = data.select_dtypes(exclude=['number']).columns
    data[non_numeric_columns] = data[non_numeric_columns].fillna(data[non_numeric_columns].mode().iloc[0])

    data = pd.get_dummies(data, columns=['Name'], drop_first=True)

    print("Processed Data is:\n", data.head())

    X = data.drop(columns=['sr']).values
    y = data['sr'].values

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=0)

    scaler = StandardScaler()
    X_train = scaler.fit_transform(X_train)
    X_test = scaler.transform(X_test)
    
    return X_train, X_test, y_train, y_test

if __name__ == "__main__":
    X_train, X_test, y_train, y_test = dataPreprocessing("C:\\Users\\Shreya\\Downloads\\IPLdata2021.csv")
