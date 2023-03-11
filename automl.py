import streamlit as st
import pandas as pd
import numpy as np
import pickle

from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score, mean_squared_error

# Function to load the saved model from pickle file
def load_model():
    with open('model.pkl', 'rb') as file:
        model = pickle.load(file)
    return model

# Function to predict the target variable using the user-input values
def predict(model, features):
    return model.predict(features)

# Function to simulate the model with random values
def simulate(model, n_samples=100):
    np.random.seed(0)
    features = np.random.randn(n_samples, len(model.coef_))
    target = predict(model, features)
    return features, target

# Function to train the linear regression model and save it to a pickle file
def train_and_save_model(df, target_col):
    X = df.drop(target_col, axis=1)
    y = df[target_col]
    model = LinearRegression().fit(X, y)
    with open('model.pkl', 'wb') as file:
        pickle.dump(model, file)
    return model

# Streamlit app code
def main():
    st.set_page_config(page_title='Linear Regression Simulator', page_icon=':bar_chart:', layout='wide')

    st.title('Linear Regression Simulator')

    # File upload feature
    file = st.file_uploader('Upload a CSV file', type=['csv'])
    if not file:
        st.warning('Please upload a CSV file')
        return

    # Read the uploaded file as a DataFrame
    df = pd.read_csv(file)

    # Column selection feature
    st.subheader('Select the target variable')
    target_col = st.selectbox('Column', df.columns)

    # Model simulation feature
    if st.button('Simulate Model'):
        try:
            # Load the saved model from pickle file
            model = load_model()
        except:
            # Train and save the model if the pickle file is not found
            st.warning('Model not found. Training a new model...')
            model = train_and_save_model(df, target_col)

        # Simulate the model with random values
        features, target = simulate(model)

        # Show the model performance metrics
        r2 = r2_score(target, predict(model, features))
        mse = mean_squared_error(target, predict(model, features))
        st.write('Model R^2:', r2)
        st.write('Model MSE:', mse)

        # Show the simulated data and predicted target variable
        st.subheader('Simulated Data')
        st.write(pd.DataFrame(features))
        st.subheader('Predicted Target')
        st.write(pd.DataFrame(target))

if __name__ == '__main__':
    main()
