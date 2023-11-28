import requests


fp = open('test.csv', 'rb')
file = {'file': fp}

response = requests.post(
    url='http://127.0.0.1:8000/predict_items',
    files=file
)

print(response.status_code)
print(response.json())
