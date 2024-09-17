import numpy as np
import pandas as pd
import streamlit as st
import plotly.express as px
import proj
import plotly.graph_objects as go
import os
from sklearn.metrics import mean_squared_error, r2_score

# Generate dataset
# Number of data points
n = 100

# for consistency
np.random.seed(42)

# generate random study hours betn 0 - 30
study_hours = np.random.randint(0, 30, n)  

# parental support: 0 - low 1 - medium 2 - high
parental_support = np.random.uniform(0,3, n).astype(int)

# Coefficients for StudyHours and ParentalSupport
alpha = 2   
beta = 10   

# Noise term
e = np.random.normal(0, 5, n)

# Generate FinalGrade based on a linear relationship with some noise
final_grade = alpha * study_hours + beta * parental_support + e

# Ensure that the final grade is between 0 and 100
final_grade = np.clip(final_grade, 0, 100)

# Create a DataFrame
data = pd.DataFrame({
    'StudyHours': study_hours,
    'ParentalSupport': parental_support,
    'FinalGrade': final_grade
})
data.to_csv('initial_data.csv',index=False)
# Display the first few rows of the dataset
data.isna().sum() # no empty data
data.info()
# data.duplicated().sum()
data.drop_duplicates() # removed duplicated rows

fig = px.scatter(data,x='StudyHours',y='FinalGrade')
fig.update_layout(
    title="StudyHours vs FinalGrade",
    xaxis_title="Study Hours",
    yaxis_title="Final Grade"
)
st.plotly_chart(fig)

data.corr()



corr_matrix = data.corr()

# Create heatmap
fig = go.Figure(data=go.Heatmap(
                z=corr_matrix.values, # value of each block
                x=corr_matrix.columns,
                y=corr_matrix.columns,
                colorscale='RdBu',
                text=corr_matrix.values, # before annotations
                texttemplate="%{text:.2f}", # annotations
                zmin=-1, zmax=1,))
st.plotly_chart(fig)

# x = input y = output
X = data.drop(columns='FinalGrade')
y = data['FinalGrade']

# Train and split
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Model training
from sklearn.linear_model import LinearRegression
model = LinearRegression()
model.fit(X_train, y_train)
y_pred = model.predict(X_test)

mse = mean_squared_error(y_test,y_pred)
r2 = r2_score(y_test,y_pred)

#print evaluation
st.write(f"Mean squared error: {mse}")
st.write(f"R-squared: {r2}")

# model.coef_, model.intercept_
coef_hours = model.coef_[0]
coef_support = model.coef_[1]
c = model.intercept_

st.write(f"The study hours coefficient is {coef_hours:.2f}")
st.write(f"The parental support coefficient is {coef_support:.2f}")
st.write(f"The intercept is {c:.2f}")

st.title("Interactive Grade Predictor")

# Input for study hours
study_hours = st.number_input("How many hours did you study?", min_value=0, max_value=40)

# Input for parental support
option_map = {"Low": 0, "Medium": 1, "High": 2}
parental_support = st.selectbox("Choose parental support", list(option_map.keys()))
selected_value = option_map[parental_support]

if 'prediction_made' not in st.session_state:
    st.session_state.prediction_made = False

if st.button('Predict Grade', key='predict_button'):
    predicted_grade = coef_hours*study_hours + coef_support * selected_value
    predicted_grade = max(0, min(100, predicted_grade))  # Ensure grade is between 0 and 100
    st.write(f"Predicted Grade: {predicted_grade:.2f}")
    st.session_state.prediction_made = True
    st.session_state.predicted_grade = predicted_grade

if st.session_state.prediction_made:
    if st.button('Save predictions?', key='save_button'):
        proj.save_prediction(study_hours, selected_value, st.session_state.predicted_grade)
        st.success("Prediction saved successfully!")
        st.session_state.prediction_made = False  # Reset for next prediction
try: 
    if os.path.getsize('saved_predictions.csv') > 0:
        if st.button('Show predictions', key='show_button'):
            proj.load_saved_predictions()
    if os.path.getsize('saved_predictions.csv') > 0:
        if st.button('Show visualization', key='show_vizual'):
            proj.generate_visualizations()
except:
    pass
