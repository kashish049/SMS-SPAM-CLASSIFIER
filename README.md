# 📩 SMS Spam Classifier — Machine Learning & NLP Project

A **classical machine learning–based SMS spam detection system** that classifies messages as **Spam** or **Ham (Not Spam)** using **Natural Language Processing (NLP)** techniques and multiple ML models.

This project demonstrates an **end-to-end ML workflow** including data preprocessing, feature extraction, model training, evaluation, and comparison.

---

## 🧠 Problem Statement
Unwanted SMS spam leads to financial fraud and poor user experience.  
The objective of this project is to **automatically detect spam messages** using machine learning and NLP techniques.

---

## 🛠️ Tech Stack
- **Language:** Python  
- **Libraries:** Scikit-learn, NLTK, NumPy, Pandas, Matplotlib  
- **Machine Learning Models:**
  - Support Vector Machine (SVM)
  - K-Nearest Neighbors (KNN)
  - Multilayer Perceptron (MLP)
- **Concepts:** NLP, Feature Engineering, Model Evaluation

---

## ✨ Key Features
- End-to-end **machine learning pipeline** for text classification
- **Text preprocessing**:
  - Cleaning & normalization
  - Tokenization
  - Stopword removal
  - Lemmatization
- Feature extraction from SMS text
- Training and comparison of multiple ML models
- Performance evaluation using:
  - Accuracy
  - Precision
  - Recall
  - F1-score
- **Confusion matrix analysis** for error inspection

---

## 📂 Project Structure
SMS-SPAM-CLASSIFIER/
│
├── README.md
├── sms_spam_classifier.ipynb # Complete ML pipeline
├── sms_spam_collection.csv # Dataset

yaml
Copy code

---

## ⚙️ How It Works

### 1. Data Preprocessing
- Removed punctuation, numbers, and extra spaces
- Converted text to lowercase
- Tokenized SMS messages
- Removed stopwords
- Applied lemmatization

### 2. Feature Engineering
- Converted processed text into numerical features suitable for ML models
- Focused on interpretable, classical NLP representations

### 3. Model Training
Trained and evaluated the following models:
- **SVM** – margin-based classifier
- **KNN** – distance-based classifier
- **MLP** – shallow neural network for comparison

### 4. Model Evaluation
- Used train–test split
- Compared models using precision–recall trade-offs
- Analyzed confusion matrices to understand misclassifications

---

## 📊 Results & Observations

| Model | Accuracy | Observation |
|------|----------|-------------|
| SVM | ~98% | Best balance between precision and recall |
| KNN | ~93% | High precision, lower recall |
| MLP | ~97% | Strong performance, slightly less interpretable |

**Key Learning:**  
Accuracy alone is not sufficient in spam detection; **precision and recall are critical** to reduce false positives.

---

## 🧪 Sample Prediction
```python
message = "Congratulations! You've won a free ticket. Call now!"
Output:

nginx
Copy code
Spam
🎯 What This Project Demonstrates
Strong understanding of NLP fundamentals

Ability to build ML systems from scratch

Experience in model evaluation and comparison

Practical use of classical machine learning models

Clean and explainable experimentation

📚 Dataset
UCI SMS Spam Collection Dataset

🚀 Future Improvements
TF-IDF and n-gram feature comparison

Hyperparameter tuning

REST API deployment

Real-time inference demo

👤 Author
Ashish Kumar

GitHub: https://github.com/kashish049

Email: kashish04945@gmail.com

⭐ Support
If you found this project useful, feel free to give it a ⭐.
