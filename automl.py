import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
import pickle

# Set page title
st.set_page_config(page_title="Linear Regression Simulator")

# Title and description
st.title("Linear Regression Simulator")
st.write("Upload a dataset, select features, and train a linear regression model. The trained model will be saved as a pickle file and can be used for predictions in the simulator section of the app.")

# Upload dataset
st.header("Upload Dataset")
data_file = st.file_uploader("Upload CSV", type=["csv"])

if data_file is not None:
    data = pd.read_csv(data_file)
    st.write(data.head())

    # Choose label column
    st.header("Choose Label Column")
    label_col = st.selectbox("Select label column:", options=data.columns)

    # Choose numerical and categorical features
    st.header("Choose Features")
    numerical_cols = data.select_dtypes(include=np.number).columns.tolist()
    categorical_cols = data.select_dtypes(include='object').columns.tolist()
    chosen_numerical_cols = st.multiselect("Select numerical features:", options=numerical_cols, default=numerical_cols)
    chosen_categorical_cols = st.multiselect("Select categorical features:", options=categorical_cols, default=categorical_cols)

    # Create X and y
    X = data[chosen_numerical_cols + chosen_categorical_cols]
    y = data[label_col]

    # Convert categorical features to dummy variables
    X = pd.get_dummies(X)

    # Split data into train and test sets
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    # Train model
    st.header("Train Model")
    model = LinearRegression()
    model.fit(X_train, y_train)

    # Save model as pickle file
    filename = 'model.pkl'
    pickle.dump(model, open(filename, 'wb'))

    # Load model from pickle file
    loaded_model = pickle.load(open(filename, 'rb'))

    # Simulator
    st.header("Simulator")

    # Select feature values
    feature_values = []
    for col in X.columns:
        if col in chosen_numerical_cols:
            value = st.number_input(f"Enter value for {col}", step=0.01)
        else:
            value = st.selectbox(f"Select value for {col}", options=X[col].unique().tolist())
        feature_values.append(value)

    # Predict using loaded model
    prediction = loaded_model.predict([feature_values])[0]

    # Display prediction
    st.write(f"Prediction: {prediction:.2f}")

    
