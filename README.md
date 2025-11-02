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
