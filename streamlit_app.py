import streamlit as st
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score

# Load the dataset
@st.cache
def load_data():
    data = pd.read_csv('AgrcultureDataset.csv')  # Replace 'your_dataset.csv' with your dataset file path
    return data

data = load_data()

# Create a Streamlit web app
st.title('Crop Profit Prediction App')
st.sidebar.title('User Input Features')

# Sidebar for user input
district_name = st.sidebar.selectbox('Select District', data['District_Name'].unique())
crop = st.sidebar.selectbox('Select Crop', data['Crop'].unique())

# Filtering the dataset based on user input
filtered_data = data[(data['District_Name'] == district_name) & (data['Crop'] == crop)]

# Display the filtered data
st.write('Filtered Data:')
st.write(filtered_data)

# Split the data into features (X) and target (y)
X = filtered_data[['Area', 'Production', 'Income', 'Expense']]
y = filtered_data['Profit']

# Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Train a linear regression model
model = LinearRegression()
model.fit(X_train, y_train)

# Prediction function
def predict_profit(area, production, income, expense):
    input_data = [[area, production, income, expense]]
    prediction = model.predict(input_data)
    return prediction[0]

# Input elements for user interaction
st.sidebar.header('Enter Input Values:')
area = st.sidebar.number_input('Area', min_value=0)
production = st.sidebar.number_input('Production', min_value=0)
income = st.sidebar.number_input('Income', min_value=0)
expense = st.sidebar.number_input('Expense', min_value=0)

# Make predictions based on user input
if st.sidebar.button('Predict'):
    predicted_profit = predict_profit(area, production, income, expense)
    st.write(f'Predicted Profit: {predicted_profit:.2f}')

# Model evaluation (optional)
st.header('Model Evaluation (Optional):')
y_pred = model.predict(X_test)
mse = mean_squared_error(y_test, y_pred)
r2 = r2_score(y_test, y_pred)
st.write(f'Mean Squared Error: {mse:.2f}')
st.write(f'R-squared (R2): {r2:.2f}')
