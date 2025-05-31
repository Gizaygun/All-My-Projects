import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.impute import SimpleImputer

# Load Titanic dataset
data = pd.read_csv('titanic.csv')

# Remove rows with missing 'Age' values
data = data.dropna(subset=['Age'])

# Replace missing values in 'Embarked' with the most frequent value
imputer = SimpleImputer(strategy='most_frequent')
data['Embarked'] = imputer.fit_transform(data[['Embarked']]).ravel()  # Flatten the result to a 1D array

# One-hot encode the 'Sex' column
data = pd.get_dummies(data, columns=['Sex'], drop_first=True)

# Standardize 'Age' and 'Fare' columns
scaler = StandardScaler()
data[['Age', 'Fare']] = scaler.fit_transform(data[['Age', 'Fare']])

# Display first few rows of the processed dataset
print(data.head())
