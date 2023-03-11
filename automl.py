import streamlit as st
import pandas as pd
import numpy as np
import pickle

# Function to load the pickled model
@st.cache(allow_output_mutation=True)
def load_model():
    with open('model.pkl', 'rb') as file:
        model = pickle.load(file)
    return model

# Function to get predictions using the loaded model
def get_predictions(model, inputs):
    return model.predict(inputs)

# Function to run the simulation
def run_simulation(model, inputs):
    # Get the predictions
    predictions = get_predictions(model, inputs)

    # Create a DataFrame with the results
    results_df = pd.DataFrame({
        'Input 1': inputs[:, 0],
        'Input 2': inputs[:, 1],
        'Prediction': predictions
    })

    # Display the results
    st.write('### Results')
    st.write(results_df)

# Main function
def main():
    # Set the page title and the page icon
    st.set_page_config(page_title='Linear Regression Simulator', page_icon=':bar_chart:')

    # Load the pickled model
    model = load_model()

    # Add a title
    st.title('Linear Regression Simulator')

    # Add a subtitle
    st.write('This app simulates a linear regression model using the pickled model.')

    # Add a file uploader
    st.write('### Upload the data')
    uploaded_file = st.file_uploader('Upload a CSV file', type='csv')

    # If the file is uploaded
    if uploaded_file is not None:
        # Load the data into a DataFrame
        data = pd.read_csv(uploaded_file)

        # Show the first 5 rows of the data
        st.write('### Data preview')
        st.write(data.head())

        # Choose the input variables
        st.write('### Choose the input variables')
        input_columns = st.multiselect('Select the input variables', data.columns)

        # Choose the target variable
        st.write('### Choose the target variable')
        target_column = st.selectbox('Select the target variable', data.columns)

        # Split the data into inputs and target
        inputs = data[input_columns].values
        target = data[target_column].values

        # Add a slider for the number of simulations
        st.write('### Choose the number of simulations')
        num_simulations = st.slider('Number of simulations', min_value=1, max_value=100, value=10)

        # Add a button to run the simulation
        run_simulation_button = st.button('Run simulation')

        # If the simulation button is clicked
        if run_simulation_button:
            # Run the simulation
            for i in range(num_simulations):
                st.write(f'## Simulation {i+1}')
                run_simulation(model, inputs)

# Run the app
if __name__ == '__main__':
    main()
