from flask import Flask
from flask_cors import CORS
from routes.text_route import bp as text_bp

app = Flask(__name__)
CORS(app)  # <-- This allows requests from frontend at localhost:3000
app.register_blueprint(text_bp, url_prefix='/api/text')

@app.get("/")
def home():
    return {"message": "Fraud detection API running"}

if __name__ == "__main__":
    app.run(debug=True)
