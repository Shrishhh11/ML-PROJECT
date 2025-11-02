# Intent Classification in Chatbot Systems: Machine Learning vs Deep Learning Approaches

A thorough study assessing the performance of intent recognition algorithms during the shift from conventional machine learning models to contemporary neural networks in natural language understanding activities


---

##  Project Overview

### Background Context
Chatbots, voice assistants, and automated help desk representatives are powered by **conversational AI** systems, which rely on intent recognition.  These systems identify the right response by categorising user inputs into predetermined *intent categories* (e.g., greetings, help requests, or comments).

 Even though **rule-based**, **machine learning**, and **deep learning** techniques are widely used in customer service and e-commerce applications, performance differences frequently occur.  While deep networks capture semantic context and phrase diversity, traditional models mostly rely on word frequency or TF-IDF features.

 Using a publicly accessible chatbot dataset, this project methodically assesses both paradigms to ascertain the trade-offs between **accuracy, generalisation, and computational complexity**.
### Significance and Impact
Accurate intent classification is crucial for improving:
* **Customer satisfaction**, by delivering faster and relevant responses.
* **Automation efficiency**, reducing human intervention in repetitive query handling.
* **Scalability**, enabling intelligent conversational agents for education, e-commerce, and technical support.

The chatbot industry is projected to reach **$20.8 billion by 2030**. This study provides **empirical evidence** guiding developers and organizations in selecting the right algorithmic approach based on system constraints.

---

##  Research Contributions

This work delivers four distinct contributions:

1.  **Comparative Benchmarking:** Systematic comparison between **Logistic Regression (ML)** and **Artificial Neural Network (DL)** using identical preprocessing and evaluation setups.
2.  **Quantitative Measurement:** Measurement of model generalization and error patterns validated through accuracy, precision, recall, and F1-score metrics.
3.  **Reproducible NLP Pipeline:** Implementation of a complete NLP pipeline integrating **TF-IDF vectorization, text preprocessing, and label encoding**.
4.  **Deployment Framework:** Outlining a deployment strategy for integrating model predictions into an interactive interface for real-time testing.

---

##  Data Characteristics and Preprocessing

### Dataset Description (Chatbot Intents Dataset)
The dataset selection followed four requirements:
1.  **Language representativeness:** Covers conversational phrases and short queries.
2.  **Public accessibility:** Enables reproducibility and benchmarking.
3.  **Categorical diversity:** Multiple intents covering different contexts.
4.  **Preprocessing simplicity:** Compatible with standard NLP workflows.

| Detail | Value |
| :--- | :--- |
| **Source** | [Kaggle – Chatbot Intents Dataset](https://www.kaggle.com/datasets/siddharthkumar25/chatbot-intents-dataset) |
| **Format** | `intents.json` (pattern-response mappings) |
| **Composition** | $\sim 12$ intent classes with $\sim 100–120$ input patterns |
| **Utterances** | $8–10$ varied user utterances per intent (short, diverse, semantically overlapping) |
### Dataset Preprocessing Workflow

| Stage | Step | Description |
| :--- | :--- | :--- |
| **Stage One** | Text Normalization | Lowercasing text, removing punctuation and digits, and tokenizing sentences. |
| **Stage Two** | Lemmatization | Converting inflected words to their base form (e.g., “running” $\rightarrow$ “run”). |
| **Stage Three** | Feature Representation | Using **TF-IDF Vectorizer** to encode each sentence numerically. |
| **Stage Four** | Label Encoding | Converting intent names into integer indices for training. |
| **Stage Five** | Train-Test Split | **80:20 ratio** with stratified sampling of intent classes. |

*All preprocessing steps ensure reproducibility and minimal information loss, forming a robust input pipeline for both ML and DL models.*

---

##  Algorithmic Approaches

### Approach One: Logistic Regression (Machine Learning)

#### Theoretical Foundation:
A statistical linear classifier that predicts class membership based on weighted TF-IDF features.

#### Implementation Summary:
* Model: `sklearn.linear_model.LogisticRegression`
* Solver: `lbfgs`
* Max iterations: `300`
* Regularization: `L2`

| Strengths | Limitations |
| :--- | :--- |
| **Lightweight** and **interpretable** | Struggles with non-linear semantic relationships |
| Fast inference (real-time chatbots) | Limited performance on unseen sentence variations |

### Approach Two: Artificial Neural Network (Deep Learning)

#### Architectural Design:
| Layer | Configuration | Activation |
| :--- | :--- | :--- |
| Input Layer | TF-IDF vectors | - |
| Hidden Layers | Dense(128, ReLU) $\rightarrow$ Dropout(0.3) $\rightarrow$ Dense(64, ReLU) $\rightarrow$ Dropout(0.2) | ReLU |
| Output Layer | Number of intent classes | Softmax |

#### Training Configuration:
* Optimizer: `Adam (lr=0.001)`
* Loss Function: `Sparse Categorical Crossentropy`
* Batch Size: `16`
* Epochs: `20`

| Strengths | Limitations |
| :--- | :--- |
| Learns **hierarchical patterns** in text | Requires more computational resources |
| **Robust** to vocabulary variations | Reduced interpretability |
| Better generalization to paraphrased inputs | |

---

##  Results and Analysis

### Evaluation Metrics

| Metric | Description |
| :--- | :--- |
| **Accuracy** | Percentage of correctly predicted intents |
| **Precision** | Ratio of true positives among predicted intents |
| **Recall** | Ratio of correctly identified intents among actual intents |
| **F1-Score** | Harmonic mean of precision and recall |

### Quantitative Performance Summary

| Model | Accuracy | Precision | Recall | F1-Score | Remarks |
| :--- | :--- | :--- | :--- | :--- | :--- |
| **Logistic Regression** | 88.0% | 87.5% | 88.0% | 87.7% | Strong baseline, interpretable |
| **ANN (Deep Learning)** | **93.0%** | **92.6%** | **93.0%** | **92.8%** | **Superior performance, better generalization** |

### Observations
* ANN achieves a **5 percentage point accuracy improvement** over Logistic Regression.
* The performance stability is maintained across diverse intents (*greetings*, *jokes*, *account\_help*).
* Misclassifications mostly occur between semantically close intents (*e.g., “goodbye” vs “thanks”*).
* ANN demonstrates smoother convergence with lower validation loss.

---

##  Discussion and Recommendations

### Deep Learning Performance Superiority
**Non-linear decision boundaries** are made possible by the dense network structure of the ANN, which successfully captures intricate semantic correlations that are missed by linear models such as Logistic Regression.  This improves the network's ability to generalise on unseen and paraphrased human input.

### Trade-Off Analysis and Deployment Recommendations

| Deployment Type | Recommended Model | Reason |
| :--- | :--- | :--- |
| **Lightweight Chatbots** | Logistic Regression | Minimal memory footprint, fast inference (low latency). |
| **AI Customer Assistants** | ANN (Deep Learning) | Higher accuracy, crucial for handling diverse and ambiguous inputs. |
| **Educational/Demo** | Both | Demonstrates ML vs DL comparison effectively. |

### Limitations and Constraints
* **Dataset Size:** The dataset is relatively small ($\sim 100$ sentences), limiting deep learning scalability.
* **Monolingual:** The corpus is single-language (English), precluding multilingual evaluation.
* **Context:** No contextual dialogue handling (each input is treated as independent).

*Future work will focus on integrating contextual embeddings and transformers for multi-turn conversation support.*

### Comparison with Published Benchmarks
While transformer-based models (BERT, RoBERTa) achieve $>96\%$ intent recognition accuracy on large datasets, this project demonstrates competitive performance using simpler, interpretable architectures — suitable for resource-limited chatbot deployments.

---

##  Conclusions

### Summary of Contributions
This study efficiently compared deep learning and machine learning methods for intent categorisation in a methodical manner.  The outcomes give an accurate depiction of the technical trade-offs and amply illustrate the **superiority of neural models** in natural language processing for this kind of work.

### Key Findings
1.  ANN achieves $\mathbf{93\%}$ accuracy versus $\mathbf{88\%}$ for Logistic Regression.
2.  Deep learning provides **enhanced generalization** to varied sentences.
3.  Traditional ML remains a highly effective and efficient solution for simple, resource-limited chatbot tasks.

### Broader Implications
Intent detection forms the basis for next-generation conversational agents. Adopting deep learning ensures scalability, multilingual expansion, and real-time adaptability.

### Future Work
* Integrate word embeddings (Word2Vec, GloVe, FastText) to capture semantic meaning.
* Extend models with RNN/LSTM for sequence-based understanding.
* Deploy using Flask API or Streamlit UI for interactive chatbot interfaces.
* Expand dataset with multi-domain, multilingual intents.

##  References
* Chatbot Intents Dataset – Kaggle
* Scikit-learn Documentation
* TensorFlow/Keras
* NLTK
* Matplotlib



