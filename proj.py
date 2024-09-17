import pandas as pd
import os
import streamlit as st
import plotly.express as px
import plotly.graph_objects as go

def save_prediction(study_hours, parental_support, predicted_grade):
    file_exists = os.path.isfile('saved_predictions.csv')
    
    with open('saved_predictions.csv', 'a') as f:
        if not file_exists:
            f.write('StudyHours,ParentalSupport,PredictedGrade\n')
        f.write(f"{study_hours},{parental_support},{predicted_grade:.2f}\n")
    print("Prediction saved successfully.")

def load_saved_predictions():
    if os.path.exists('saved_predictions.csv'):
        saved_data = pd.read_csv('saved_predictions.csv')
        st.write("\nSaved Predictions:")
        st.write(saved_data)
    else:
        st.write("No saved predictions yet.")

def generate_visualizations():
    # Load the latest data from saved_predictions.csv
    if not os.path.exists('saved_predictions.csv'):
        st.write("No saved predictions available for visualization.")
        return
    
    user_data = pd.read_csv('saved_predictions.csv')
    
    st.write("\nGenerating Data Visualizations for Latest Data...")
    

# Create heatmap
    corr_matrix = user_data.corr()
    fig = go.Figure(data=go.Heatmap(
                    z=corr_matrix.values, # value of each block
                    x=corr_matrix.columns,
                    y=corr_matrix.columns,
                    colorscale='RdBu',
                    text=corr_matrix.values, # before annotations
                    texttemplate="%{text:.2f}", # annotations
                    zmin=-1, zmax=1,))
    st.plotly_chart(fig)
    





