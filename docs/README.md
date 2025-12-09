# Fraud/Spam Text Detection

## What it Does

This project implements a text-based fraud and spam detection system that processes user messages and predicts whether they are "spam" or "ham" (legitimate). The system includes a PyTorch-based custom neural network, sklearn baseline models for comparison, data augmentation techniques, and a deployed web interface for interactive usage. Users can input text messages through either a REST API or web UI and receive predictions along with summary features such as character count, token count, and first tokens.

## Quick Start

**Backend Setup**
```bash
cd backend
python -m venv venv
source venv/bin/activate      # On Windows: venv\Scripts\activate
pip install -r requirements.txt
python app.py
```
The backend API will run at `http://127.0.0.1:5000`

**Frontend Setup**

Navigate to the frontend folder and serve `index.html` using your preferred method. Open your browser at `http://127.0.0.1:3000/` to access the UI.

**Testing**

Use `test_api.py` or the frontend interface to send text messages and view predictions and summary features.

## Video Links

- **Demo Video**: [Insert demo link here]
- **Technical Walkthrough**: [Insert walkthrough link here]

## Evaluation

**Sklearn Baselines:**
- DummyClassifier (most frequent): Accuracy: 0.6025, F1: 0.0
- LogisticRegression (balanced): Accuracy: 0.9435, F1: 0.9287

**PyTorch MLP Classifier (AdamW):**
- Test Accuracy: ~0.9565
- Test F1: ~0.9444
**PyTorch MLP Classifier (Adam):**
- Test Accuracy: ~0.9601
- Test F1: ~0.9501
**PyTorch MLP Classifier (SGD):**
- Test Accuracy: ~0.9430
- Test F1: ~0.9261

**Additional Analysis:**
- Error Analysis: Misclassified messages logged to CSV with qualitative examples
- Optimizer Comparison: Trained models with SGD, Adam, and AdamW, observing differences in validation loss and F1 scores

## Individual Contributions

**Caleb Donaldson**: All project components including data preprocessing, augmentation, PyTorch model design, optimizer comparison, error analysis, and web app deployment. Project completed individually (solo).