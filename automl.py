import streamlit as st
import pandas as pd
from sklearn.linear_model import TweedieRegressor
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split

# Function to train the model and return the R-squared score
def train_model(df, target_col):
    # Split dataset into features and target
    X = df.drop(target_col, axis=1)
    y = df[target_col]
    
    # Split dataset into training and testing sets
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    
    # Create pipeline with scaling and regression model
    pipeline = make_pipeline(StandardScaler(), TweedieRegressor(power=1, alpha=0.5, link='log'))
    
    # Fit pipeline to training data
    pipeline.fit(X_train, y_train)
    
    # Calculate and return R-squared score on testing data
    score = pipeline.score(X_test, y_test)
    return score, pipeline

# Streamlit app
def main():
    # Set page title
    st.set_page_config(page_title="Generalized Linear Regression")
    
    # Add title and description
    st.title("Generalized Linear Regression")
    st.write("Upload a CSV file and select the target variable to perform generalized linear regression on the data.")
    
    # Add file uploader and target variable selector
    uploaded_file = st.file_uploader("Upload CSV file", type="csv")
    if uploaded_file is not None:
        df = pd.read_csv(uploaded_file)
        target_col = st.selectbox("Select target variable", options=df.columns)
        
        # Train the model and show R-squared score
        try:
            score, model = train_model(df, target_col)
            st.write(f"R-squared score: {score:.2f}")
        except Exception as e:
            st.write(f"Error: {e}")
        
        # Add sliders for model parameters and predict button
        st.write("Use the sliders to adjust the model parameters, then click 'Predict' to make a prediction.")
        power = st.slider("Power", min_value=0.1, max_value=2.0, value=1.0, step=0.1)
        alpha = st.slider("Alpha", min_value=0.1, max_value=1.0, value=0.5, step=0.1)
        link = st.selectbox("Link", options=["auto", "log", "identity", "sqrt", "inverse"])
        predict_button = st.button("Predict")
        
        # Predict on new data using the model and user-selected parameters
        if predict_button:
            new_data = {}
            for col in df.columns:
                if col != target_col:
                    value = st.number_input(col)
                    new_data[col] = [value]
            new_df = pd.DataFrame(new_data)
            prediction = model.predict(new_df)
            st.write(f"Prediction: {prediction[0]:.2f}")
        
if __name__ == '__main__':
    main()
