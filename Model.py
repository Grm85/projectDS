
import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import confusion_matrix, classification_report
import os

# Function to load and preprocess data
def load_and_preprocess_data(file_path, sheet_name='vw_ChurnData'):
    # Check if file exists
    assert os.path.isfile(file_path), "File not found"
    
    # Load data
    data = pd.read_excel(file_path, sheet_name=sheet_name)

    # Drop columns that won't be used for prediction
    data = data.drop(['Customer_ID', 'Churn_Category', 'Churn_Reason'], axis=1)

    # List of columns to encode
    columns_to_encode = [
        'Gender', 'Married', 'State', 'Value_Deal', 'Phone_Service', 'Multiple_Lines',
        'Internet_Service', 'Internet_Type', 'Online_Security', 'Online_Backup',
        'Device_Protection_Plan', 'Premium_Support', 'Streaming_TV', 'Streaming_Movies',
        'Streaming_Music', 'Unlimited_Data', 'Contract', 'Paperless_Billing',
        'Payment_Method'
    ]

    # Encode categorical variables
    label_encoders = {}
    for column in columns_to_encode:
        label_encoders[column] = LabelEncoder()
        data[column] = label_encoders[column].fit_transform(data[column])

    # Manually encode the target variable 'Customer_Status'
    data['Customer_Status'] = data['Customer_Status'].map({'Stayed': 0, 'Churned': 1})

    return data, label_encoders

# Function to split data
def split_data(data):
    # Split features and target variable
    X = data.drop('Customer_Status', axis=1)
    y = data['Customer_Status']

    # Split data into training and testing sets
    return train_test_split(X, y, test_size=0.2, random_state=42)

# Function to train Random Forest model
def train_model(X_train, y_train):
    rf_model = RandomForestClassifier(n_estimators=100, random_state=42)
    rf_model.fit(X_train, y_train)
    return rf_model

# Function to evaluate model
def evaluate_model(model, X_test, y_test):
    y_pred = model.predict(X_test)
    
    # Print confusion matrix and classification report
    print("Confusion Matrix:")
    print(confusion_matrix(y_test, y_pred))
    
    print("\nClassification Report:")
    print(classification_report(y_test, y_pred))
    
    return y_pred

# Function to get feature importance
def get_feature_importance(model, X):
    importances = model.feature_importances_
    indices = np.argsort(importances)[::-1]
    
    return importances, X.columns[indices]

# Function to make predictions on new data
def predict_new_data(model, new_data, label_encoders, file_path):
    # Load new data
    new_data = pd.read_excel(file_path, sheet_name='vw_JoinData')
    original_data = new_data.copy()

    # Retain Customer_ID
    customer_ids = new_data['Customer_ID']

    # Drop unnecessary columns
    new_data = new_data.drop(['Customer_ID', 'Customer_Status', 'Churn_Category', 'Churn_Reason'], axis=1)

    # Encode categorical variables using saved label encoders
    for column in new_data.select_dtypes(include=['object']).columns:
        new_data[column] = label_encoders[column].transform(new_data[column])

    # Make predictions
    new_predictions = model.predict(new_data)

    # Add predictions to the original DataFrame
    original_data['Customer_Status_Predicted'] = new_predictions

    # Filter the DataFrame for churned customers
    churned_customers = original_data[original_data['Customer_Status_Predicted'] == 1]

    # Save to CSV
    churned_customers.to_csv(r"C:\Users\LENOVO\OneDrive\Desktop\ChurnAnalysis\Predictions.csv", index=False)

    return churned_customers

# Example of using the functions
if __name__ == "__main__":
    # File path
    file_path = r"C:\Users\LENOVO\OneDrive\Desktop\ChurnAnalysis\Prediction_Data.xlsx"
    
    # Load and preprocess data
    data, label_encoders = load_and_preprocess_data(file_path)
    
    # Split data into training and testing sets
    X_train, X_test, y_train, y_test = split_data(data)
    
    # Train the Random Forest model
    model = train_model(X_train, y_train)
    
    # Evaluate the model
    evaluate_model(model, X_test, y_test)
    
    # Get and print feature importance
    importances, feature_names = get_feature_importance(model, X_train)
    print("Feature Importance:")
    for feature, importance in zip(feature_names, importances):
        print(f"{feature}: {importance}")
    
    # Predict on new data
    churned_customers = predict_new_data(model, data, label_encoders, file_path)
    print("Predicted churned customers saved to Predictions.csv")
