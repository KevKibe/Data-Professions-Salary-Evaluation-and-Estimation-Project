import requests 

input_features = {
    'employee_residence': 'Europe',
    'company_location': 'Europe',
    'experience_level': 'EX',
    'employment_type': 'FL',
    'job_title': 'data scientist',
    'company_size': 'L',
    'year': 2023
}

payload = {
    "input_features": input_features
}

response = requests.post('http://127.0.0.1:5000/predict', json=payload)

if response.status_code == 200:
    data = response.json()
    prediction = data['prediction']
    print("Estimated Salary:", prediction)
else:
    print("Error:", response.text)