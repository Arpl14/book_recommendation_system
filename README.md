# 📚 Book Genre Classification and Recommendation System

This project aims to build a comprehensive machine learning pipeline to classify books into genres based on their descriptions and provide intelligent, personalized book recommendations. We also integrated a Large Language Model (LLM) chatbot interface to enhance user interaction and real-time recommendations.

---

## 🔍 Project Motivation

Manual genre classification in physical libraries is often inconsistent and subjective. Our goal is to automate this process using Natural Language Processing (NLP) and Machine Learning (ML), and augment the experience with an interactive recommendation system powered by LLMs. This improves accuracy, consistency, and user engagement in discovering relevant books.

---

## 🧠 Technologies Used

- **Python**, **Pandas**, **Scikit-learn**, **NLTK**
- **Machine Learning Models**: Logistic Regression, Random Forest, SVM, XGBoost, Gradient Boosting, HistGradientBoosting
- **NLP**: TF-IDF, Stemming, Lemmatization, Stopword Removal
- **Dimensionality Reduction**: PCA, t-SNE
- **Resampling**: SMOTE for handling class imbalance
- **Deep Learning**: LSTM with TensorFlow/Keras
- **Chatbot Interface**: Streamlit + OpenAI's GPT via API

---

## 📈 Stepwise Project Pipeline

### ✅ 1. Data Acquisition and Cleaning
- Dataset: Goodreads dataset from Kaggle.
- Removed rows with missing book descriptions or genres.
- Cleaned HTML tags, punctuation, and special characters from text.
- Normalized text: lowercasing, removing stopwords, and lemmatization.
- Merged rare genres into broader categories and removed genres with <30 samples.

### 📊 2. Exploratory Data Analysis (EDA)
- Visualized genre distribution, word clouds, and rating histograms.
- Identified dominant genres and imbalances in the dataset.

### ✍️ 3. Feature Engineering
- Extracted features using **TF-IDF** vectorization.
- Reduced dimensionality using **PCA** (95% variance retained).
- Visualized clusters with **t-SNE** (used only for interpretation, not modeling).

### 🤖 4. Genre Classification Modeling
Trained and evaluated multiple models:
- **Logistic Regression** – Baseline model for genre prediction.
- **Random Forest** – High accuracy but prone to overfitting.
- **XGBoost** – Fast and effective on sparse TF-IDF features.
- **HistGradientBoostingClassifier** – Best generalization performance.
- **LSTM Neural Network** – Explored deep learning on tokenized padded sequences.

➡️ **Final Model Chosen**: `HistGradientBoostingClassifier`  
Due to:
- Strong performance on imbalanced data
- Robust generalization (verified via learning & validation curves)
- Less overfitting compared to other models

### 🔧 5. Hyperparameter Tuning
- Used **GridSearchCV** and **RandomizedSearchCV** for tuning:
  - Random Forest
  - XGBoost
  - HistGradientBoostingClassifier

### 🤝 6. Recommendation System Integration
- Built a **content-based recommender** using cosine similarity on TF-IDF features.
- Enhanced with an **LLM-powered chatbot interface** (via Streamlit + OpenAI API).
  - Users can ask for books by author, similar books, or book ratings.
  - Natural conversational queries supported.

---

## 📊 Model Evaluation

- **Accuracy**: Evaluated across models on a stratified test set.
- **Metrics**: Precision, Recall, F1-Score, Confusion Matrix.
- **Resampling**: SMOTE helped balance minority genres during training.
- **Model Export**: Final models saved using Pickle for deployment.

Conclusion: This project builds an intelligent system that classifies book genres and provides personalized recommendations using machine learning and NLP. Combining TF-IDF, ensemble models, and an LLM-powered chatbot results in a scalable solution that enhances user experience and streamlines book discovery through automation and conversational AI.

