from flask import Flask, request, jsonify
import pickle
import numpy as np
import pandas as pd

app = Flask(__name__)

model = pickle.load(open('sal_model_V4.pkl', 'rb'))

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
    
    df = pd.DataFrame([feature_values])
    df_encoded = pd.DataFrame(columns=feature_names)
    
    for feature_name in feature_names:
        feature_value = feature_values.get(feature_name.split('_')[0])
        
        df_encoded[feature_name] = [int(feature_value == feature_name.split('_')[1])]
    input_data = df_encoded.values
    
    prediction = model.predict(input_data)
    ranges = [(15000, 33066),
              (33066, 51133),
              (51133, 69200),
              (69200, 87266),
              (87266, 105333),
              (105333, 123400),
              (123400, 141466),
              (141466, 159533),
              (159533, 177600),
              (177600, 195666),
              (195666, 213733),
              (213733, 231800),
              (231800, 249866),
              (249866, 267933),
              (267933, 286000)]
    
    prediction_range = None
    for range_min, range_max in ranges:
        if range_min <= prediction < range_max:
            prediction_range = f"{range_min:,} - {range_max:,}"
            break
    
    return prediction_range


@app.route('/predict', methods=['POST'])
def predict():
    data = request.get_json(force=True)
    input_features = data['input_features']
    prediction = make_prediction(input_features)
    return jsonify({'prediction': prediction })


if __name__ == '__main__':
    app.run(debug=True)
