# Breast Cancer Classification Using Machine Learning

## 1. Problem Statement

The main objective of this project is to predict whether a breast tumor is Benign or Malignant using Machine Learning classification algorithms.

Breast cancer detection is an important classification problem because early identification of malignant cases can help in further diagnosis and treatment.

In this project, I have implemented five different Machine Learning classification models on the same dataset. After training the models, I compared their performance using different evaluation metrics to find which model gives the best overall result.

---

## 2. Dataset Description

**Dataset Name:** Breast Cancer Wisconsin Diagnostic Dataset

**Source:** Kaggle

The dataset contains different measurements of breast mass samples which are used to identify whether the tumor is Benign or Malignant.

The dataset contains:

- Total Records: 569
- Input Features: 30
- Target Column: `diagnosis`
- Problem Type: Binary Classification

### Target Values

- `0` = Benign
- `1` = Malignant

The `id` column was removed because it is only used for identification and does not help the Machine Learning model in prediction.

The dataset was divided into:

- 80% Training Data
- 20% Testing Data

I used stratified train-test splitting so that the distribution of Benign and Malignant cases remains similar in both training and testing data.

Feature scaling using `StandardScaler` was also applied for the models where scaling is useful.

---

## 3. Machine Learning Models Used

I implemented the following five classification models:

1. Logistic Regression
2. Decision Tree Classifier
3. K-Nearest Neighbors (KNN)
4. Gaussian Naive Bayes
5. Random Forest Classifier

Random Forest was used as the Ensemble Learning model.

---

## 4. Evaluation Metrics

To compare the performance of all models, I used the following evaluation metrics:

- Accuracy
- AUC Score
- Precision
- Recall
- F1 Score
- Matthews Correlation Coefficient (MCC)

Using multiple evaluation metrics gives a better understanding of model performance instead of depending only on Accuracy.

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

Logistic Regression performed very well on the dataset. It achieved an Accuracy of 96.49% and the highest AUC score of 99.60%.

The model also achieved 97.50% Precision and 92.86% Recall. Overall, Logistic Regression gave very good and balanced classification results.

### Decision Tree

Decision Tree achieved an Accuracy of 92.98%.

Its Precision, Recall and F1 Score were around 90.48%. The model performed reasonably well, but its overall performance was lower compared to Logistic Regression, KNN and Random Forest.

### K-Nearest Neighbors (KNN)

KNN achieved an Accuracy of 95.61% and Precision of 97.44%.

Its AUC score was 98.23%, which shows good classification performance.

Feature scaling was important for KNN because KNN works by calculating the distance between data points.

### Gaussian Naive Bayes

Naive Bayes achieved an Accuracy of 92.11% and a good AUC score of 98.91%.

However, its Recall was 85.71%, which was the lowest among all five models.

This means Naive Bayes missed more Malignant cases compared to the other models.

### Random Forest

Random Forest gave the best overall performance.

It achieved the highest Accuracy of 97.37% and a Precision of 100%.

It also achieved:

- Recall: 92.86%
- F1 Score: 96.30%
- MCC: 0.9442
- AUC: 99.29%

Random Forest performed strongly across almost all the evaluation metrics.

---

## 7. Overall Winner

### Random Forest Classifier

Based on the overall comparison, I selected **Random Forest as the best-performing model**.

Random Forest achieved:

- Accuracy: 97.37%
- AUC: 99.29%
- Precision: 100%
- Recall: 92.86%
- F1 Score: 96.30%
- MCC: 0.9442

Logistic Regression achieved a slightly higher AUC score of 99.60%.

However, Random Forest achieved the highest Accuracy, Precision, F1 Score and MCC among all the models.

Therefore, based on the overall performance across different evaluation metrics, **Random Forest was selected as the final winner**.

---

## 8. Streamlit Application

I developed an interactive Streamlit application to test and compare the trained Machine Learning models.

The application provides the following features:

- Upload test dataset in CSV format
- Select a Machine Learning model from the dropdown
- Evaluate the selected model
- Display Accuracy
- Display AUC Score
- Display Precision
- Display Recall
- Display F1 Score
- Display MCC
- Display Confusion Matrix
- Display Classification Report

The user can select any of the five trained models and check its performance on the uploaded test dataset.

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

Add GitHub repository link here.

---

## 11. Live Streamlit Application

Streamlit Application Link:

Add deployed Streamlit application link here.
