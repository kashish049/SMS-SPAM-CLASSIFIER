📩 SMS Spam Classifier — Machine Learning & NLP Project

A classical machine learning–based SMS spam detection system built to classify messages as Spam or Ham (Not Spam) using natural language preprocessing and multiple ML models.
This project demonstrates end-to-end ML workflow, including data cleaning, feature extraction, model training, evaluation, and comparison.

📌 Resume-aligned focus: NLP preprocessing, ML model training, precision–recall analysis, and evaluation — no black-box deep learning.

🧠 Problem Statement

SMS spam causes financial fraud and poor user experience.
The goal of this project is to automatically classify SMS messages using machine learning and NLP techniques to reduce spam exposure.

🛠️ Tech Stack

Language: Python

Libraries: Scikit-learn, NLTK, NumPy, Pandas, Matplotlib

ML Models:

Support Vector Machine (SVM)

K-Nearest Neighbors (KNN)

Multilayer Perceptron (MLP)

Concepts: NLP, Feature Engineering, Model Evaluation

✨ Key Features

End-to-end ML pipeline for text classification

Text preprocessing:

Cleaning & normalization

Tokenization

Stopword removal

Lemmatization

Feature extraction using word-based representations

Multiple model training & comparison

Performance evaluation using:

Accuracy

Precision

Recall

F1-score

Confusion matrix analysis to study false positives/negatives

📂 Project Structure
SMS-SPAM-CLASSIFIER/
│
├── README.md                   # Project documentation
├── sms_spam_classifier.ipynb   # Complete ML pipeline (Colab notebook)
├── sms_spam_collection.csv     # Dataset


📌 Entire implementation is available in a single notebook for clarity and easy review by recruiters.

⚙️ How It Works (Step-by-Step)
1️⃣ Data Preprocessing

Removed punctuation, numbers, and extra spaces

Converted text to lowercase

Tokenized SMS messages

Removed stopwords

Applied lemmatization for word normalization

2️⃣ Feature Engineering

Converted cleaned text into numerical vectors suitable for ML models

Focused on interpretable, classical NLP features

3️⃣ Model Training

Trained and compared:

SVM → strong margin-based classifier

KNN → distance-based baseline model

MLP → shallow neural network for comparison

4️⃣ Model Evaluation

Used train–test split

Evaluated using precision–recall trade-offs

Analyzed confusion matrices to understand misclassifications

📊 Results & Observations
Model	Accuracy	Key Insight
SVM	~98%	Best balance of precision & recall
KNN	~93%	High precision, lower recall
MLP	~97%	Strong performance, slightly less interpretable

📌 Learning Outcome:
Accuracy alone is insufficient — precision and recall matter more in spam detection to avoid false positives.

🧪 Sample Prediction
message = "Congratulations! You've won a free ticket. Call now!"


Output:

Spam

🎯 What This Project Demonstrates (For Recruiters)

Strong understanding of NLP fundamentals

Ability to build ML systems from scratch

Experience with model evaluation and trade-off analysis

Clear grasp of classical ML models (often preferred in interviews)

Clean, explainable, and reproducible experimentation

📚 Dataset

UCI SMS Spam Collection Dataset
Widely used benchmark dataset for NLP and spam detection tasks.

🚀 Future Improvements

Add TF-IDF + n-grams comparison

Hyperparameter tuning

Deploy as a REST API

Add real-time inference demo

👤 Author

Ashish Kumar

GitHub: https://github.com/kashish049

Email: kashish04945@gmail.com
