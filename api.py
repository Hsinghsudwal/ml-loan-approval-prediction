import requests

url = 'http://localhost:9696/predict'

input = {
    "no_of_dependents": 3,
    "education":"Not Graduate",
    "self_employed":"No",
    "income_annum":9800000,
    "loan_amount":21200000,
    "loan_term":15,
    "cibil_score":355,
    "residential_assets_value":22000000,
    "commercial_assets_value":8900000,
    "luxury_assets_value":31800000,
    "bank_asset_value":14400000,
}

response = requests.post(url, json=input).json()
print(response)