import streamlit as st
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score


def get_dataset(file):
    dataset = pd.read_csv(file)
    return dataset


def get_features(dataset):
    numerical_features = list(dataset.select_dtypes(include=['int', 'float']).columns)
    categorical_features = list(dataset.select_dtypes(include=['object']).columns)
    return numerical_features, categorical_features


def encode_categorical_features(dataset, categorical_features):
    encoder = LabelEncoder()
    for feature in categorical_features:
        dataset[feature] = encoder.fit_transform(dataset[feature])
    return dataset


def train_model(dataset, numerical_features, categorical_features, label_column):
    X = dataset[numerical_features + categorical_features]
    y = dataset[label_column]

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)

    rf = RandomForestClassifier()
    rf.fit(X_train, y_train)

    y_pred = rf.predict(X_test)
    accuracy = accuracy_score(y_test, y_pred)

    return rf, accuracy


def predict(model, numerical_features, categorical_features, inputs):
    inputs = pd.DataFrame([inputs], columns=numerical_features + categorical_features)
    prediction = model.predict(inputs)[0]
    return prediction


def simulate(model, numerical_features, categorical_features):
    st.subheader("Simulator")
    inputs = {}
    for feature in numerical_features:
        inputs[feature] = st.slider(f"{feature}:", float(dataset[feature].min()), float(dataset[feature].max()))

    for feature in categorical_features:
        inputs[feature] = st.selectbox(f"{feature}:", list(dataset[feature].unique()))

    prediction = predict(model, numerical_features, categorical_features, inputs)
    st.write("Prediction:", prediction)


# Main function
def main():
    st.title("Machine Learning App")
    file = st.file_uploader("Upload Dataset", type=["csv"])
    if not file:
        st.warning("Please upload a CSV file.")
        return

    dataset = get_dataset(file)

    st.subheader("Data")

    label_column = st.selectbox("Select Label Column", list(dataset.columns))
    numerical_features, categorical_features = get_features(dataset)
    dataset = encode_categorical_features(dataset, categorical_features)

    st.write(dataset)

    if len(numerical_features) == 0 or len(categorical_features) == 0 or not label_column:
        st.warning("Please choose appropriate columns for features and label.")
        return

    model, accuracy = train_model(dataset, numerical_features, categorical_features, label_column)

    st.subheader("Model Performance")
    st.write(f"Accuracy: {accuracy:.2f}")

    simulate(model, numerical_features, categorical_features)


if __name__ == '__main__':
    main()
