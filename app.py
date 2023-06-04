import streamlit as st
from google.cloud import storage
import pickle
import numpy as np
import os 




def load_model():
    with open('sal_model_V4.pkl', 'rb') as file:
        model = pickle.load(file)
    return model
model=load_model()
def make_prediction(feature_values):
    feature_names = ['employee_residence_Africa',
                     'employee_residence_Asia', 'employee_residence_Europe',
                     'employee_residence_North America', 'employee_residence_Oceania',
                     'employee_residence_South America', 'company_location_Africa',
                     'company_location_Asia', 'company_location_Europe',
                     'company_location_North America', 'company_location_Oceania',
                     'company_location_South America',
                     'experience_level_EN', 
                     'experience_level_EX', 
                     'experience_level_MI', 
                     'experience_level_SE',
                     'employment_type_CT', 
                     'employment_type_FL', 
                     'employment_type_FT', 
                     'employment_type_PT', 
                     'job_title_data engineer',
                     'job_title_data analyst', 
                     'job_title_data scientist',
                     'job_title_machine learning engineer', 
                     'company_size_M', 
                     'company_size_S', 
                     'company_size_L',
                     'year_2020',
                     'year_2021', 
                     'year_2022',
                     'year_2023']
    # Create a numpy array for the input data
    input_data = np.array([[feature_values['employee_residence'] == 'Africa',
                            feature_values['employee_residence'] == 'Asia',
                            feature_values['employee_residence'] == 'Europe',
                            feature_values['employee_residence'] == 'North America',
                            feature_values['employee_residence'] == 'Oceania',
                            feature_values['employee_residence'] == 'South America',
                            feature_values['company_location'] == 'Africa',
                            feature_values['company_location'] == 'Asia',
                            feature_values['company_location'] == 'Europe',
                            feature_values['company_location'] == 'North America',
                            feature_values['company_location'] == 'Oceania',
                            feature_values['company_location'] == 'South America',
                            feature_values['experience_level'] == 'EN',
                            feature_values['experience_level'] == 'EX',
                            feature_values['experience_level'] == 'MI',
                            feature_values['experience_level'] == 'SE',
                            feature_values['employment_type'] == 'CT', 
                            feature_values['employment_type'] == 'FL', 
                            feature_values['employment_type'] == 'FT', 
                            feature_values['employment_type'] == 'PT', 
                            feature_values['job_title'] == 'data analyst', 
                            feature_values['job_title'] == 'data engineer',
                            feature_values['job_title'] == 'data scientist', 
                            feature_values['job_title'] == 'machine learning engineer',
                            feature_values['company_size'] == 'M',
                            feature_values['company_size'] =='S',
                            feature_values['company_size'] =='L',
                            feature_values['year'] == 2020,
                            feature_values['year'] == 2021,
                            feature_values['year'] == 2022,
                            feature_values['year'] == 2023]])
    prediction = model.predict(input_data)
    ranges = [(15000 , 33066),
              (33066 , 51133),
              (51133 , 69200),
              (69200 , 87266),
              (87266 , 105333),
              (105333 , 123400),
              (123400 , 141466),
              (141466 , 159533),
              (159533 , 177600),
              (177600 , 195666),
              (195666 , 213733),
              (213733 , 231800),
              (231800 , 249866),
              (249866 , 267933),
              (267933 , 286000)]
    prediction_range = None
    for range_min, range_max in ranges:
        if range_min <= prediction < range_max:
            prediction_range = f"{range_min:,} - {range_max:,}"
            break
    return prediction_range
st.title("Data Professions Salary Estimation")
input_features = {}

input_features['experience_level'] = st.selectbox('Experience Level:', options=[('EN','Entry-level / Junior'),('MI','Mid-level / Intermediate'),('SE','Senior-level / Expert'),('EX','Executive-level / Director')], format_func=lambda x: x[1], key='experience_level')
input_features['employment_type'] = st.selectbox('Employment Type:', options=[('PT','Part-time'),('FT','Full-time'),('CT','Contract'),('FL','Freelance')], format_func=lambda x: x[1], key='employment_type')
input_features['job_title'] = st.selectbox('Job Title:', options=['data analyst','data engineer','data scientist','machine learning engineer'], key='job_title')
input_features['year'] = 2023
input_features['company_size'] = st.selectbox('Company Size:', options=[('S','less than 50 employees (small)'),('M','50 to 250 employees (medium)'),('L','more than 250 employees (large)')], format_func=lambda x: x[1], key='company_size')
input_features['employee_residence'] = st.selectbox('Employee Residence:', options=[('Africa','Africa'),('Asia','Asia'),('Europe','Europe'),('North America','North America'),('Oceania','Oceania'),('South America','South America')], format_func=lambda x: x[1], key='employee_residence')
input_features['company_location'] = st.selectbox('Company Location:', options=[('Africa','Africa'),('Asia','Asia'),('Europe','Europe'),('North America','North America'),('Oceania','Oceania'),('South America','South America')], format_func=lambda x: x[1], key='company_location')

if st.button('Estimate'):
    prediction = make_prediction(input_features)
    st.write("Estimation in USD :", prediction)
