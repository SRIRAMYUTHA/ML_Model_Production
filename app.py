import streamlit as st
import numpy as np
from pickle import load
from sklearn.neighbors import KNeighborsClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC 
from sklearn.tree import DecisionTreeClassifier 
from sklearn.ensemble import RandomForestClassifier 
import warnings 
warnings.filterwarnings("ignore")
# Loading pretrained classifiers from pickle file

scaler = load(open('models/standard_scaler.pkl', 'rb'))
person_home_ownership_encoder=load(open('models/person_home_ownership.pkl', 'rb'))
loan_intent_encoder=load(open('models/loan_intent.pkl', 'rb'))
loan_grade_encoder=load(open('models/loan_grade.pkl', 'rb'))
cb_person_default_on_file_encoder=load(open('models/cb_person_default_on_file.pkl', 'rb'))

#knn_classifier = load(open('models/knn_model.pkl', 'rb'))
#lr_classifier = load(open('models/lr_model.pkl', 'rb'))
#dt_classifier = load(open('models/dt_model.pkl', 'rb'))
#sv_classifier = load(open('models/sv_model.pkl', 'rb'))
rf_classifier = load(open('models/rf_model.pkl', 'rb'))

## Taking Input From User
age = st.number_input("Enter Person Age",format="%.2f")
income=st.number_input(" Enter Income",format="%.2f")
emp_length=st.number_input("Enter Person Employee Length",format="%.2f")
loan_amt=st.number_input("Enter Loan Amount",format="%.2f")
loan_int_rate=st.number_input("Enter Loan Interest Rate",format="%.2f")
loan_percent_income=st.number_input("Enter Loan Percent Income:",format="%.2f")
cb_person_cred_hist_length=st.number_input("Enter Credit History Length:",format="%.2f")


person_home_ownership=st.text_input("Enter Person Home Ownership:")
loan_intent=st.text_input("Enter Loan Intent:")
loan_grade=st.text_input("Enter Loan Grade:")
cb_person_default_on_file=st.text_input(" Enter Historic Default:")

btn_click = st.button("Predict")

query_point = np.array([age,income,emp_length,loan_amt,loan_int_rate,loan_percent_income,cb_person_cred_hist_length]).reshape(1, -1)
query_point_transformed = scaler.transform(query_point)

person_home_ownership_transformed=0
for i in person_home_ownership_encoder:
    if i in person_home_ownership:
        #print(person_home_ownership,person_home_ownership_encoder[i])
        person_home_ownership_transformed=person_home_ownership_encoder[i]

loan_intent_transformed=0
for i in loan_intent_encoder:
    if i in loan_intent:
        #print(loan_intent,loan_intent_encoder[i])
        loan_intent_transformed=loan_intent_encoder[i]

loan_grade_transformed=0
for i in loan_grade_encoder:
    if i in loan_grade:
        #print(loan_grade,loan_grade_encoder[i])
        loan_grade_transformed=loan_grade_encoder[i]

cb_person_default_on_file_transformed=0
for i in cb_person_default_on_file_encoder:
    if i in cb_person_default_on_file:
        #print(cb_person_default_on_file,cb_person_default_on_file_encoder[i])
        cb_person_default_on_file_transformed=cb_person_default_on_file_encoder[i]

#query_point_2 = np.array([person_home_ownership_transformed[0],loan_intent_transformed[0],loan_grade_transformed[0],cb_person_default_on_file_transformed[0]]).reshape(1, -1)

query_point_2 = np.array([person_home_ownership_transformed,
                         loan_intent_transformed,
                         loan_grade_transformed,
                         cb_person_default_on_file_transformed]).reshape(1, -1)
## Prediction
if btn_click == True:
    if age and income and emp_length and loan_amt and loan_int_rate and loan_percent_income and cb_person_cred_hist_length and person_home_ownership and loan_intent and loan_grade and cb_person_default_on_file_transformed:
        new_query_point=np.append(query_point,query_point_2).reshape(1,-1)
        
        pred = rf_classifier.predict(new_query_point)
        if pred==0:
            st.success("Non Default")
            
        else:
            st.success("Default")
            
        
    else:
        st.error("Enter the values properly.")