import streamlit as st
import pandas as pd
import seaborn as sns
import numpy as np

from sklearn.preprocessing import LabelEncoder
le_sex = LabelEncoder()

@st.cache_data
def load_data():

    diabetes = pd.read_csv("diabetes.csv")
    diabetes.drop('smoking_history', axis=1, inplace=True)
    diabetes['gender'] = le_sex.fit_transform(diabetes['gender'])
    # Calculate the first quartile (Q1) and third quartile (Q3)
    Q1 = np.percentile(diabetes['bmi'], 25)
    Q3 = np.percentile(diabetes['bmi'], 75)

    # Calculate the interquartile range (IQR)
    IQR = Q3 - Q1

    # Define the outlier bounds
    lower_bound = Q1 - 1.5 * IQR
    upper_bound = Q3 + 1.5 * IQR

    # Filter the DataFrame to remove outliers
    diabetes_cleaned = diabetes[(diabetes['bmi'] >= lower_bound) & (diabetes['bmi'] <= upper_bound)]

    # Print the shape of the cleaned DataFrame to see how many outliers were removed
    return diabetes_cleaned

df = load_data()

def show_explore_page():
    st.title("Explore Diabetes Data")

    st.write(

        ###Diabetes from keggle dataset
    )
    
    heatmap = sns.heatmap(df[['diabetes','age','bmi','HbA1c_level','blood_glucose_level','gender','heart_disease']].corr(),annot = True)
    st.write("Heatmap of the data")
    st.pyplot(heatmap.figure)