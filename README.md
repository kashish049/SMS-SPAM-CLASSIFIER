

# 📧 SMS Spam Classifier with Skip-Gram Embeddings & Weighted Voting

A robust **SMS spam detection system** that leverages **skip-gram embeddings** for feature extraction and combines **multiple machine learning models** (SVM, KNN, MLP) with a **weighted voting mechanism** for highly accurate predictions.

---

## 🚀 Features
- **Skip-gram embeddings** to capture meaningful word patterns from SMS text.
- Uses **three models**:
  - **Linear SVM**  
  - **K-Nearest Neighbors (KNN)**  
  - **Multilayer Perceptron (MLP)**  
- **Weighted voting system** that prioritizes higher-performing models.
- **Preprocessing pipeline**: tokenization, stopword removal, and lemmatization.
- **Confusion matrix visualization** for performance evaluation.

---

## 📝 Table of Contents
1. [Installation](#installation)  
2. [Usage](#usage)  
3. [Project Structure](#project-structure)  
4. [How It Works](#how-it-works)  
5. [Results](#results)  
6. [Contributing](#contributing)  
7. [License](#license)  

---

## ⚙️ Installation
1. Clone this repository:
   ```bash
   git clone https://github.com/kashish049/sms-spam-classifier.git
   cd sms-spam-classifier
   ```

2. Install the required libraries:
   ```bash
   pip install -r requirements.txt
   ```

3. (Optional) Download NLTK resources:
   ```python
   import nltk
   nltk.download('stopwords')
   nltk.download('wordnet')
   ```

---

## 🎯 Usage
1. **Train Models**:
   ```bash
   python train.py
   ```

2. **Predict New Messages**:
   ```python
   from main import predict_with_weighted_voting, models, weights

   message = "Your package will be delivered tomorrow. Thank you for shopping with us!"
   prediction = predict_with_weighted_voting(models, message, weights)
   print(f"Prediction: {prediction}")
   ```

3. **Evaluate Model Performance**:
   ```bash
   python evaluate.py
   ```

---

## 📁 Project Structure
```
sms-spam-classifier/
│
├── README.md               # Project documentation
├── requirements.txt        # List of dependencies
├── train.py                # Script to train all models
├── evaluate.py             # Script to evaluate model performance
├── main.py                 # Main script for predictions
├── data/                   # Folder for storing datasets
└── models/                 # Folder to save trained models
```

---

## 💡 How It Works
1. **Data Preprocessing**:
   - Removes special characters and numbers.
   - Tokenizes messages and removes stopwords.
   - Lemmatizes words for normalization.

2. **Feature Engineering**:
   - Uses **skip-gram embeddings** to generate 1-gram and 2-gram word patterns.

3. **Model Training**:
   - Trains **SVM, KNN, and MLP** models on the preprocessed SMS data.

4. **Weighted Voting**:
   - Each model is assigned a **weight based on accuracy**.
   - **Weighted votes** are used to decide the final prediction (Spam/Ham).

---

## 📊 Results

### **Performance Metrics**

| **Model** | **Accuracy** | **Precision** | **Recall** | **F1-Score** |
|-----------|--------------|---------------|------------|--------------|
| **SVM**   | 98.0%        | 0.99          | 0.85       | 0.92         |
| **KNN**   | 93.3%        | 1.00          | 0.49       | 0.66         |
| **MLP**   | 97.6%        | 1.00          | 0.82       | 0.90         |

### **Confusion Matrices**
#### SVM Model:
```
[[1452    1]
 [  32  187]]
```
#### KNN Model:
```
[[1453    0]
 [ 112  107]]
```
#### MLP Model:
```
[[1453    0]
 [  40  179]]
```

---

### **Prediction Example:**
```python
message = "Congratulations! You've won a free ticket. Call now!"
prediction = predict_with_weighted_voting(models, message, weights)
print(f"Prediction: {prediction}")
```

**Expected Output:**
```
Individual Model Predictions: {'SVM': 'Spam', 'KNN': 'Spam', 'MLP': 'Spam'}
Final Consensus Prediction: Spam
```

---

## 📈 Confusion Matrix Visualization

Below is a sample **confusion matrix** for all three models.

![Confusion Matrix](assets/confusion_matrix.png)

---

## 🤝 Contributing
We welcome contributions! If you'd like to improve the model or add new features:
1. Fork the repository.
2. Create a new branch:
   ```bash
   git checkout -b feature-branch
   ```
3. Commit your changes:
   ```bash
   git commit -m 'Add new feature'
   ```
4. Push to your branch:
   ```bash
   git push origin feature-branch
   ```
5. Open a pull request.

---

## 📄 License
This project is not licensed 

---

## 💬 Contact
For questions or suggestions, feel free to reach out at:  
**Email:** ashishkumarnith4@gmail.com  
**GitHub:** [kashish049](https://github.com/kashish049)

---

## ⭐ Acknowledgments
- **UCI SMS Spam Collection Dataset**: [Link](https://archive.ics.uci.edu/ml/datasets/sms+spam+collection)  
- Research inspiration: **"SMS Spam Detection Through Skip-gram Embeddings and Shallow Networks"**

---

## 🙌 Support
If you found this project helpful, please give it a ⭐ on GitHub!

---

This updated **README.md** now reflects the **actual model performance metrics** and maintains a professional structure, ready to be used for your GitHub project. Let me know if you need further changes!
