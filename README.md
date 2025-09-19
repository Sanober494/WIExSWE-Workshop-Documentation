# Breast Cancer Prediction with Machine Learning


Breast cancer is a significant health challenge worldwide, affecting millions of women annually. Early detection and diagnosis are crucial in increasing the chances of successful treatment and survival. This project aims to develop a machine learning model that can accurately predict whether breast tumors are benign (non-cancerous) or malignant (cancerous) based on medical data. By leveraging predictive modeling techniques, this solution will assist healthcare professionals in diagnosing breast cancer early, improving patient outcomes, and potentially saving lives.

## Table of Contents
1. [Project Objective](#project-objective)
2. [Dataset Information](#dataset-information)
3. [Data Preprocessing](#data-preprocessing)
4. [Model Development and Methodology](#model-development-and-methodology)
5. [Evaluation Metrics](#evaluation-metrics)
6. [Usage](#usage)
7. [Conclusion and Impact](#conclusion-and-impact)

---

## Project Objective

The primary objective of this project is to develop a highly accurate and reliable machine learning model to predict breast tumor classification—whether a tumor is benign or malignant. The ultimate goal is to provide healthcare professionals with an effective tool that aids in early cancer detection, thereby facilitating timely medical interventions and improving the quality of patient care. By analyzing key tumor characteristics, this model serves as a critical decision-support tool for oncologists and healthcare providers.

---

## Dataset Information

We utilized the **Breast Cancer Wisconsin (Diagnostic) Dataset**, which contains crucial medical features derived from breast cancer biopsy samples. This dataset is widely used in the medical field for breast cancer research and prediction modeling.

- **Features**: The dataset includes 30 features, such as:
  - **Tumor size**
  - **Cell shape**
  - **Texture and smoothness of cells**
  - **Perimeter and area of the tumor**
  - **Mitotic count** (related to the number of cell divisions)

Each instance in the dataset is labeled as either **benign (B)** or **malignant (M)**, providing the basis for supervised learning models.

| Feature | Description |
|---------|-------------|
| Radius | Mean of distances from the center to points on the tumor perimeter |
| Texture | Standard deviation of gray-scale values |
| Perimeter | Measurement of the tumor perimeter |
| Area | Measurement of the tumor area |
| Smoothness | Variability in the smoothness of tumor edges |
| **Diagnosis** | **Target variable: Benign (B) or Malignant (M)** |


---

## Data Preprocessing

To ensure that the machine learning models perform optimally, comprehensive data preprocessing steps were applied to the raw dataset. These steps are crucial to remove inconsistencies, normalize the data, and prepare it for model training. Key steps include:

1. **Handling Missing Values**: We performed imputation or removal of missing data to avoid any gaps that could lead to inaccuracies in predictions.
2. **Feature Scaling**: Standardization or normalization of numerical features was performed using **Min-Max scaling** to ensure uniformity across the feature set, preventing any bias during model training.
3. **Encoding Categorical Variables**: The target variable (Diagnosis) was label-encoded, converting benign (B) and malignant (M) into numerical values, allowing the models to interpret the target correctly.
4. **Train-Test Split**: The dataset was split into a training set (80%) and a test set (20%) to evaluate the model’s generalization to unseen data.

---

## Model Development and Methodology

We experimented with multiple machine learning models to ensure high accuracy and robustness in our predictions. The following algorithms were employed and evaluated:

1. **Logistic Regression (Tuned)**: This was the final model selected due to its superior performance in terms of recall, precision, and overall balanced accuracy.
2. **Support Vector Machine (SVM)**: Explored due to its effectiveness in binary classification problems.
3. **Decision Tree**: A non-parametric model that works well for classification tasks with interpretable decision-making.
4. **Random Forest**: An ensemble method known for reducing variance and improving accuracy.
5. **Naive Bayes**: A simple probabilistic model that performed moderately well in this classification task.
6. **XGBoost**: A powerful boosting algorithm often used in structured data competitions and real-world applications.

---

## Evaluation Metrics

For this project, several key evaluation metrics were considered to ensure the model's effectiveness:

- **Accuracy**: The proportion of correctly classified instances over the total instances. However, in medical diagnoses, accuracy alone isn’t enough.
- **Precision**: The ratio of true positive predictions to all positive predictions. It helps in determining how many of the predicted positive cases are actually malignant.
- **Recall (Sensitivity)**: A critical metric for this project, recall is the ability of the model to correctly identify malignant cases, which is vital to minimize false negatives (missed cancer cases).
- **F1-Score**: The harmonic mean of precision and recall, providing a balance between the two.
- **ROC-AUC**: This metric measures the model’s ability to differentiate between benign and malignant cases across different threshold values, showing how well the model performs regardless of class distribution.

---



## Usage

For Beginners: 
 ### Use Google Collab ✨
1. Download this repository in your local machine
2. This includes the .ipynb and the csv data file.
3. Upload the .ipynb in Google Collab, along with the csv data file.
4. Click Run cell-by-cell in order to understand whats happening.
5. Happy Learning Cupcakes!

For Users with VS code/ Editors and pre installed necessary libraries:
 ### Clone this Repo ✨
 
```bash
# Create a virtual Environment, its a good practice :)
python -m venv tutorial-env

# Clone the repository
git clone https://github.com/Sanober494/WIExSWE-Workshop-Documentation.git

#Install dependencies if needed

# Run the notebook
jupyter notebook Code.ipynb

# Happy Learning Sugars :)
```
---


## Conclusion and Impact

The development of our Breast Cancer Prediction Model underscores the growing importance of machine learning in healthcare, especially in the early detection of cancer. With its high accuracy and recall, this model empowers healthcare professionals to diagnose breast cancer early, allowing for timely treatment that can dramatically improve survival rates.

By leveraging data-driven techniques, this project illustrates how artificial intelligence can elevate medical diagnostics. The early identification of malignant tumors not only opens doors to more effective treatments but also gives patients a better chance at recovery. Every decision in healthcare carries immense weight, and this model demonstrates the critical role precision can play in saving lives.

In conclusion, this project highlights how machine learning can reshape the future of healthcare, particularly in the fight against breast cancer. As AI technology continues to evolve, integrating models like this into real-world medical workflows will help raise the quality of care, offering hope and better outcomes for countless patients.

---
