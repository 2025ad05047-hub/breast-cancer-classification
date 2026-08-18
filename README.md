# Breast Cancer Classification Using Machine Learning

## 1. Problem Statement

Breast cancer is one of the most common types of cancer, and early detection is important for effective treatment. The objective of this project is to develop and compare multiple machine learning classification models for predicting whether a breast tumor is benign or malignant based on diagnostic measurements.

Five machine learning classification algorithms are implemented and evaluated on the same dataset. Their performance is compared using multiple evaluation metrics.

---

## 2. Dataset Description

**Dataset:** Breast Cancer Wisconsin Diagnostic Dataset

**Source:** Kaggle / Wisconsin Diagnostic Breast Cancer Dataset

The dataset contains diagnostic measurements of breast mass samples. Each sample is classified as either benign or malignant.

- Total instances: 569
- Predictive features used: 30
- Target variable: `diagnosis`
- Classification type: Binary Classification

### Target Classes

- `0` = Benign
- `1` = Malignant

The `id` column was removed because it is only an identifier and does not provide useful information for classification.

The dataset was divided into:

- 80% Training Data
- 20% Testing Data

A stratified train-test split was used to preserve the class distribution.

StandardScaler was applied for models that benefit from feature scaling.

---

## 3. Machine Learning Models Used

The following five classification models were implemented:

1. Logistic Regression
2. Decision Tree Classifier
3. K-Nearest Neighbors (KNN)
4. Gaussian Naive Bayes
5. Random Forest Classifier

Random Forest was used as the ensemble learning model.

---

## 4. Evaluation Metrics

Each model was evaluated using the following metrics:

- Accuracy
- AUC Score
- Precision
- Recall
- F1 Score
- Matthews Correlation Coefficient (MCC)

---

## 5. Model Performance Comparison

| ML Model | Accuracy | AUC | Precision | Recall | F1 | MCC |
|---|---:|---:|---:|---:|---:|---:|
| Logistic Regression | 0.9649 | 0.9960 | 0.9750 | 0.9286 | 0.9512 | 0.9245 |
| Decision Tree | 0.9298 | 0.9246 | 0.9048 | 0.9048 | 0.9048 | 0.8492 |
| KNN | 0.9561 | 0.9823 | 0.9744 | 0.9048 | 0.9383 | 0.9058 |
| Naive Bayes | 0.9211 | 0.9891 | 0.9231 | 0.8571 | 0.8889 | 0.8292 |
| Random Forest | 0.9737 | 0.9929 | 1.0000 | 0.9286 | 0.9630 | 0.9442 |

---

## 6. Model Observations

### Logistic Regression

Logistic Regression achieved an accuracy of 96.49% and the highest AUC score of 99.60%. It also achieved 97.50% precision and 92.86% recall. The results indicate that Logistic Regression performs very well for this binary classification problem.

### Decision Tree

Decision Tree achieved an accuracy of 92.98%. Its precision, recall and F1 score were approximately 90.48%. Although the model performed reasonably well, its overall performance was lower than Logistic Regression, KNN and Random Forest.

### K-Nearest Neighbors (KNN)

KNN achieved an accuracy of 95.61% with a precision of 97.44%. Its AUC score was 98.23%. Feature scaling was important for KNN because the algorithm determines classes based on distances between observations.

### Gaussian Naive Bayes

Naive Bayes achieved an accuracy of 92.11% and an AUC score of 98.91%. However, its recall was 85.71%, which was the lowest among the evaluated models. This indicates that Naive Bayes missed more malignant cases than the other models.

### Random Forest

Random Forest achieved the highest accuracy of 97.37%. It achieved 100% precision, a recall of 92.86%, an F1 score of 96.30%, and the highest MCC value of 0.9442. It showed strong overall performance across the evaluation metrics.

---

## 7. Overall Winner

**Random Forest Classifier**

Random Forest was selected as the overall best-performing model.

It achieved:

- Accuracy: 97.37%
- AUC: 99.29%
- Precision: 100%
- Recall: 92.86%
- F1 Score: 96.30%
- MCC: 0.9442

Although Logistic Regression achieved a slightly higher AUC score, Random Forest achieved the highest accuracy, precision, F1 score and MCC. Therefore, Random Forest demonstrated the strongest overall classification performance.

---

## 8. Streamlit Application

An interactive Streamlit application was developed for evaluating the trained machine learning models.

The application provides:

- CSV test dataset upload
- Model selection using a dropdown
- Evaluation of the selected model
- Accuracy
- AUC
- Precision
- Recall
- F1 Score
- MCC
- Confusion Matrix
- Classification Report

The application allows users to compare the performance of the five trained classification models on the test dataset.

---

## 9. Project Structure

ML-Assignment-2/

    app.py
    breast_cancer_classification.ipynb
    test_data.csv
    requirements.txt
    README.md

    model/
        logistic_regression.pkl
        decision_tree.pkl
        knn.pkl
        naive_bayes.pkl
        random_forest.pkl
        scaler.pkl

---

## 10. GitHub Repository

GitHub Repository Link:

To be added after uploading the project to GitHub.

---

## 11. Live Streamlit Application

Streamlit Application Link:

To be added after deployment on Streamlit Community Cloud.