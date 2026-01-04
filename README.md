# 📩 SMS Spam Classifier — Machine Learning & NLP Project

A **classical machine learning–based SMS spam detection system** that classifies messages as **Spam** or **Ham (Not Spam)** using **Natural Language Processing (NLP)** techniques and multiple ML models.

This project demonstrates an **end-to-end ML workflow** including data preprocessing, feature extraction, model training, evaluation, and comparison.

---

## 🧠 Problem Statement

Unwanted SMS spam leads to financial fraud, privacy invasion, and poor user experience.  
This project aims to **automatically detect spam messages** using machine learning and NLP techniques, providing a reliable classification system that can distinguish between legitimate and spam messages.

---

## 🛠️ Tech Stack

- **Language:** Python 3.x
- **Core Libraries:** 
  - Scikit-learn
  - NLTK
  - NumPy
  - Pandas
  - Matplotlib / Seaborn
- **Machine Learning Models:**
  - Support Vector Machine (SVM)
  - K-Nearest Neighbors (KNN)
  - Multilayer Perceptron (MLP)
- **Concepts Applied:** NLP, Feature Engineering, Model Evaluation, Text Preprocessing

---

## ✨ Key Features

- End-to-end **machine learning pipeline** for text classification
- **Comprehensive text preprocessing**:
  - Cleaning & normalization
  - Tokenization
  - Stopword removal
  - Lemmatization
- Feature extraction from SMS text using classical NLP methods
- Training and comparison of multiple ML models
- Performance evaluation using:
  - Accuracy
  - Precision
  - Recall
  - F1-score
- **Confusion matrix analysis** for error inspection and model interpretation

---

## 📂 Project Structure

```
SMS-SPAM-CLASSIFIER/
│
├── README.md
├── sms_spam_classifier.ipynb       # Complete ML pipeline (Jupyter Notebook)
├── sms_spam_collection.csv         # Dataset
└── assets/                         # (Optional) Folder for images/charts
```

---

## ⚙️ How It Works

### 1. Data Preprocessing
- Removed punctuation, numbers, and extra spaces
- Converted text to lowercase
- Tokenized SMS messages
- Removed stopwords using NLTK's stopwords corpus
- Applied lemmatization for word normalization

### 2. Feature Engineering
- Converted processed text into numerical features using CountVectorizer/TF-IDF
- Created interpretable, classical NLP representations suitable for ML models

### 3. Model Training
Trained and evaluated the following models with train-test split:
- **SVM** – margin-based classifier with linear kernel
- **KNN** – distance-based classifier with optimized k-value
- **MLP** – shallow neural network for performance comparison

### 4. Model Evaluation
- Used stratified train-test split (typically 80-20 or 70-30)
- Compared models using precision-recall trade-offs
- Analyzed confusion matrices to understand misclassification patterns

---

## 📊 Results & Observations

| Model | Accuracy | Precision | Recall | F1-Score | Observation |
|-------|----------|-----------|--------|----------|-------------|
| SVM | ~98% | ~97% | ~94% | ~95.5% | Best balance between precision and recall |
| KNN | ~93% | ~95% | ~85% | ~89.5% | High precision, lower recall |
| MLP | ~97% | ~96% | ~92% | ~94% | Strong performance, slightly less interpretable |

**Key Insight:**  
Accuracy alone is insufficient in spam detection; **precision and recall are critical** to minimize false positives (legitimate messages marked as spam) and false negatives (spam messages missed).

---

## 🧪 Sample Prediction

```python
# Example usage
message = "Congratulations! You've won a free ticket. Call now!"
# After preprocessing and model prediction:
OUTPUT: "spam"
```

---

## 🎯 What This Project Demonstrates

- Strong understanding of **NLP fundamentals** and text preprocessing
- Ability to build **end-to-end ML systems** from scratch
- Experience in **model evaluation and comparison** with proper metrics
- Practical application of **classical machine learning models** to real-world problems
- Clean, documented, and reproducible experimentation

---

## 📚 Dataset

**Source:** UCI Machine Learning Repository - SMS Spam Collection Dataset  
**Description:** A collection of 5,574 SMS messages tagged as either spam or ham (non-spam).  
**Format:** CSV with two columns: `label` (ham/spam) and `message` (text content).

---

## 🚀 Future Improvements

- Experiment with **TF-IDF and n-gram** feature variations
- Implement **hyperparameter tuning** using GridSearchCV/RandomizedSearchCV
- Build a **REST API** using Flask/FastAPI for model deployment
- Create a **real-time web interface** for demo purposes
- Explore **deep learning approaches** (LSTM, Transformers) for comparison

---

## 👤 Author

**Ashish Kumar**  
- GitHub: [https://github.com/kashish049](https://github.com/kashish049)  
- Email: kashish04945@gmail.com

---

## ⭐ Support

If you found this project useful or interesting, please consider giving it a ⭐ on GitHub!

---

## 📄 License

This project is available for educational and personal use. Please attribute the author if reused or modified.
