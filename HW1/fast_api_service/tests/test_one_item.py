import requests


with open('test.JSON') as file:
    body = file.read()

response = requests.post(
    'http://127.0.0.1:8000/predict_item',
    body
)
print(response.status_code)
print(response.json())
