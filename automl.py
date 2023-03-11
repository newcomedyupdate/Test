import streamlit as st
import pandas as pd
import pickle
from sklearn.preprocessing import StandardScaler

# Define the input form for the model simulator
def input_form(data, feature_names, feature_types, scaler):
    st.write("Enter values for the following features to predict the label.")
    input_values = {}
    for i, feature_name in enumerate(feature_names):
        st.write(f"**{feature_name}** ({feature_types[i]})")
        if feature_types[i] == 'numerical':
            input_values[feature_name] = st.number_input("", value=0.0, step=0.01)
        elif feature_types[i] == 'categorical':
            unique_values = data[feature_name].unique()
            input_values[feature_name] = st.selectbox("", unique_values)
        elif feature_types[i] == 'date':
            input_values[feature_name] = st.date_input("", value=None, min_value=None, max_value=None)
    # Convert the input values to a pandas DataFrame and scale the numerical features
    input_data = pd.DataFrame([input_values])
    for i, feature_type in enumerate(feature_types):
        if feature_type == 'numerical':
            input_data[feature_names[i]] = scaler.transform(input_data[[feature_names[i]]])
    return input_data

# Load the saved model and scaler
with open('model.pkl', 'rb') as file:
    model = pickle.load(file)
with open('scaler.pkl', 'rb') as file:
    scaler = pickle.load(file)

# Define the main function for the Streamlit app
def main():
    # Upload a dataset
    st.title("Model Simulator")
    uploaded_file = st.file_uploader("Choose a CSV file", type="csv")
    if uploaded_file is not None:
        data = pd.read_csv(uploaded_file)

        # Select the feature and label columns
        st.write("Select the feature and label columns.")
        feature_names = st.multiselect("Select feature columns", data.columns)
        label_name = st.selectbox("Select label column", data.columns)
        feature_types = []
        for feature_name in feature_names:
            if data[feature_name].dtype == 'float64' or data[feature_name].dtype == 'int64':
                feature_types.append('numerical')
            elif data[feature_name].dtype == 'object':
                feature_types.append('categorical')
            elif data[feature_name].dtype == 'datetime64':
                feature_types.append('date')

        # Make sure all necessary columns are selected
        if len(feature_names) == 0:
            st.warning("Please select at least one feature column.")
        elif label_name == '':
            st.warning("Please select a label column.")
        elif label_name not in data.columns:
            st.warning("Label column not found in dataset.")
        else:
            # Scale the numerical features
            scaler.fit(data[feature_names])
            data[feature_names] = scaler.transform(data[feature_names])

            # Define the input form for the model simulator
            input_data = input_form(data, feature_names, feature_types, scaler)

            # Make a prediction using the model
            predicted_label = model.predict(input_data)[0]

            # Display the predicted label to the user
            st.subheader("Predicted Label")
            st.write(predicted_label)

# Run the Streamlit app
if __name__ == '__main__':
    main()
