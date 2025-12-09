import requests

url = "http://127.0.0.1:5000/api/text/predict_text"

messages = [
    "Congratulations, you won a free gift card!",
    "Hey, are we still meeting tomorrow?",
    "URGENT: Your account has been compromised!",
    "Don't miss our exclusive offer today",
    "Can you send me the report by EOD?"
]

for msg in messages:
    response = requests.post(url, json={"text": msg})
    if response.status_code == 200:
        print(f"Message: {msg}")
        print("Prediction:", response.json()["prediction"])
        print("Probability:", response.json()["probability"])
        print("Summary features:", response.json()["summary_features"])
        print("-" * 50)
    else:
        print(f"Error for message: {msg}, Status Code: {response.status_code}")


