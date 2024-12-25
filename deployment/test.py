import requests

url = 'http://localhost:9696/predict'

input = {
    "no_of_dependents": 3,
    "education":"Graduate",
    "self_employed":"No",
    "income_annum":8200000,
    "loan_amount":30700000,
    "loan_term":8,
    "cibil_score":467,
    "residential_assets_value":18200000,
    "commercial_assets_value":3300000,
    "luxury_assets_value":23300000,
    "bank_asset_value":7900000,
    # loan_status: Rejected
}

response = requests.post(url, json=input).json()
print(response)