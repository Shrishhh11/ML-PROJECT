# 🧠 Intent Detection using Machine Learning and Deep Learning

---

## 📘 Description

This project focuses on **Intent Detection**, a crucial part of Natural Language Processing (NLP) systems used in chatbots and virtual assistants.  
The main goal is to classify user inputs (sentences or questions) into specific *intents* such as greetings, account help, order tracking, or jokes.

Intent classification helps chatbots understand the user’s purpose and respond meaningfully.  
In this project, we build and compare **two models**:
1. A **Machine Learning (Logistic Regression)** model  
2. A **Deep Learning (Artificial Neural Network)** model  

Both models are trained on a text-based **Chatbot Intents Dataset** and evaluated on their ability to correctly identify user intents.

---

## 📊 Dataset Source

- **Dataset:** [Chatbot Intents Dataset – Kaggle](https://www.kaggle.com/datasets/siddharthkumar25/chatbot-intents-dataset)
- **Type:** Text classification (Intent recognition)
- **Format:** `intents.json`

### 🧾 Example Data
```json
{
  "tag": "greeting",
  "patterns": ["Hi", "Hello", "Hey there", "Good morning"],
  "responses": ["Hello!", "Hi there! How can I help you?"]
}

📊 Data Overview

Total intents: ~10–12

Total text samples: ~100–120

Each intent contains multiple patterns (user queries) and corresponding responses.

🧩 Methods

This notebook implements and compares two NLP approaches for intent classification.

🧠 1️⃣ Logistic Regression (Machine Learning)

Approach:

Preprocess text (cleaning, tokenization, lemmatization)

Convert text to vectors using TF-IDF Vectorizer

Train Logistic Regression classifier to predict intent labels
from sklearn.linear_model import LogisticRegression
model_lr = LogisticRegression(max_iter=300)
model_lr.fit(X_train_vec, y_train)

