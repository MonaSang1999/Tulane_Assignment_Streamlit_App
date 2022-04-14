import streamlit as st
import pandas as pd
import numpy as np
import assignment1
from assignment1 import build_pipeline_1, load_prepare, build_pipeline_2, build_pipeline_final

st.header('Modeling and Analytics for heart_disease data')

page_selected = st.sidebar.radio("Menu", ["Home", "Model"])


if page_selected == "Home":
    st.subheader("Introduction")
    st.write("The source of 'heart_csv' file is from Kaggle website, which including 11 clinical features for predicting heart disease events. People with cardiovascular disease or who are at high cardiovascular risk (due to the presence of one or more risk factors) need early detection and management wherein a machine learning model can be of great help. ")
    st.subheader("Original source")
    df=pd.read_csv("heart.csv")
    st.dataframe(df)
    st.subheader("Predicted Result")
    option= st.selectbox(
        'Which model would you like to use?',
        ('Logistic Regression','KNeighbors Classifier','Decision Tree'))
    st.write("Your Option is", option, ":sunglasses:")
    if option == "Logistic Regression":
        if st.button("Run Logistic Regression"):
            X, y = assignment1.load_prepare()
            training_accuracy1, confusion_matrix, pipeline = assignment1.build_pipeline_1(X, y)
            col1, col2 = st.columns(2)
            col1.metric("training_accuracy", training_accuracy1)
            with col2:
                st.write("confusion_matrix")
                st.dataframe(confusion_matrix)
    if option == "KNeighbors Classifier":
        if st.button("Run KNeighbors Classifier"):
            X, y = assignment1.load_prepare()
            training_accuracy2, confusion_matrix, pipeline = assignment1.build_pipeline_2(X, y)
            col1, col2 = st.columns(2)
            col1.metric("training_accuracy", training_accuracy2)
            with col2:
                st.write("confusion_matrix")
                st.dataframe(confusion_matrix)
    if option == "Decision Tree":
        if st.button("Run Decision Tree"):
            X, y = assignment1.load_prepare()
            training_accuracy3, confusion_matrix, pipeline = assignment1.build_pipeline_final(X, y)
            col1, col2 = st.columns(2)
            col1.metric("training_accuracy", training_accuracy3)
            with col2:
                st.write("confusion_matrix")
                st.dataframe(confusion_matrix)
    st.subheader("Result")
    st.write("After apply different machine learning models based on this dataset, overall, logistic regression has the most precise accuracy to predict the occurence of heart disease from multiple factors selected ")

if page_selected == "Model":
    st.subheader("Data Components")
    selection = st.selectbox('Which data component detail would you like to see?',
        ['Age','Sex','ChestPainType','RestingBP','Cholesterol','FastingBS','RestingECG','MaxHR','ExerciseAngina','Oldpeak','ST_Slope','HeartDisease'])
    if selection == 'Age':
        st.write('age of the patient [years]')
    if selection == 'Sex':
        st.write('sex of the patient [M: Male, F: Female]')
    if selection == 'ChestPainType':
        st.write('chest pain type [TA: Typical Angina, ATA: Atypical Angina, NAP: Non-Anginal Pain, ASY: Asymptomatic]')
    if selection == 'RestingBP':
        st.write('resting blood pressure [mm Hg]')    
    if selection == 'Cholesterol':
        st.write('serum cholesterol [mm/dl]')
    if selection == 'FastingBS':
        st.write('fasting blood sugar [1: if FastingBS > 120 mg/dl, 0: otherwise]')
    if selection == 'RestingECG':
        st.write('resting electrocardiogram results [Normal: Normal, ST: having ST-T wave abnormality (T wave inversions and/or ST elevation or depression of > 0.05 mV), LVH: showing probable or definite left ventricular hypertrophy by Estes criteria')
    if selection == 'MaxHR':
        st.write('maximum heart rate achieved [Numeric value between 60 and 202]')
    if selection == 'ExerciseAngina':
        st.write('exercise-induced angina [Y: Yes, N: No]')
    if selection == 'Oldpeak':
        st.write('oldpeak = ST [Numeric value measured in depression]')
    if selection == 'ST_Slope':
        st.write('the slope of the peak exercise ST segment [Up: upsloping, Flat: flat, Down: downsloping]')
    if selection == 'HeartDisease':
        st.write('output class [1: heart disease, 0: Normal] ')

    st.subheader("Model Description")
    st.write("1. Import the dataset and set the structure for the home page")
    st.write("2. Make selectbox and run three machine learning models based on pipeline created calculating the accuracy score and confusion matrix")
    st.write("3. Show the result and make conclusion based on the model applied")
