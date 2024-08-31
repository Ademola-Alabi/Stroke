import streamlit as st
import pandas as pd
import tensorflow as tf
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.compose import ColumnTransformer

# Load the pre-trained model
model = tf.keras.models.load_model('stroke_prediction_model.h5')

# Define preprocessing function
def preprocess_input(input_data, preprocessor):
    # Convert the input data to a DataFrame
    input_df = pd.DataFrame([input_data])
    
    # Apply the preprocessing
    processed_input = preprocessor.transform(input_df)
    return processed_input

# Load the preprocessor (you'll need to recreate it with the same settings used in training)
categorical_features = ['gender', 'ever_married', 'work_type', 'Residence_type', 'smoking_status']
preprocessor = ColumnTransformer(
    transformers=[
        ('cat', OneHotEncoder(drop='first'), categorical_features)],
    remainder='passthrough')

# Fit the preprocessor with a sample dataset
sample_data = pd.read_csv('healthcare-dataset-stroke-data.csv')
X_sample = sample_data.drop(columns=['id', 'stroke'])
preprocessor.fit(X_sample)

# Streamlit app
st.title("Stroke Prediction App")

# User input
input_data = {
    'gender': st.selectbox('Gender', ['Male', 'Female']),
    'age': st.slider('Age', 1, 100),
    'hypertension': st.selectbox('Hypertension', [0, 1]),  # 0: No, 1: Yes
    'heart_disease': st.selectbox('Heart Disease', [0, 1]),  # 0: No, 1: Yes
    'ever_married': st.selectbox('Ever Married', ['No', 'Yes']),
    'work_type': st.selectbox('Work Type', ['children', 'Govt_job', 'Never_worked', 'Private', 'Self-employed']),
    'Residence_type': st.selectbox('Residence Type', ['Rural', 'Urban']),
    'avg_glucose_level': st.number_input('Average Glucose Level', 50.0, 300.0, step=0.1),
    'bmi': st.number_input('BMI', 10.0, 50.0, step=0.1),
    'smoking_status': st.selectbox('Smoking Status', ['formerly smoked', 'never smoked', 'smokes'])
}

# Preprocess the input
processed_input = preprocess_input(input_data, preprocessor)

# Display the shape of processed input
st.write(f"Processed input shape: {processed_input.shape}")

# Make a prediction using the model
if st.button("Predict"):
    prediction_prob = model.predict(processed_input)
    prediction = (prediction_prob > 0.5).astype(int)
    
    # Display the prediction
    st.write(f"Prediction: {'Stroke' if prediction == 1 else 'No Stroke'}, Probability: {prediction_prob[0][0]:.4f}")