#!/usr/bin/env python
# coding: utf-8

# ETL process




# Import necessary libraries
 
import numpy as np
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
import csv

with open('C:\Users\Admin\Desktop\Nikita Project\OPTIMIZATION  MODEL\Customer_Churn_Prediction.csv', mode='r') as file:
    csv_reader = csv.reader(file)
    header = next(csv_reader)
    data = [row for row in csv_reader]
    import csv

data = [
    ['Name', 'Age', 'City'],
    ['Alice', '23', 'Pune'],
    ['Bob', '30', 'Mumbai'],
    ['Charlie', '29', 'Delhi']
]

with open('output.csv', mode='w', newline='') as file:
    csv_writer = csv.writer(file)
    csv_writer.writerows(data)


print("Header:", header)
print("Data:", data)


# Load datasets
df = pd.read_csv('Customer_churn_prediction.csv')

# Define preprocessing for numerical features
categorical_features = ['gender', 'SeniorCitizen', 'Partner', 'Dependents', 'PhoneService', 'MultipleLines',
                        'InternetService', 'OnlineSecurity', 'OnlineBackup', 'DeviceProtection', 'TechSupport',
                        'StreamingTV', 'StreamingMovies', 'Contract', 'PaperlessBilling', 'PaymentMethod']
categorical_transformer = Pipeline(steps=[
    ('imputer', SimpleImputer(strategy='most_frequent')),
    ('onehot', OneHotEncoder(handle_unknown='ignore'))])

# Combine preprocessing steps
preprocessor = ColumnTransformer(
    transformers=[
        ('num', numerical_transformer, numerical_features),
        ('cat', categorical_transformer, categorical_features)])

# Define the pipeline 
Pipeline = Pipeline(step=[('preprocessor', preprocessor)])

# Fit the pipeline to the data
pipeline.fit(df)

# Transform the data
df_preprocessed = pipeline.transform(df)

# Convert the transformed data to a DataFrame
df_preprocessed = pd.DataFrame(df_preprocessed, columns=pipeline.get_feature_names_out())

# Save the preprocessed data to a new CSV file
df_preprocessed.to_csv('customer_churn_preprocessed.csv', index=False)

print('ETL process completed successfully!')






