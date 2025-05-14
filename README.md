https://colab.research.google.com/drive/1-YCbY9AfJJGBXCHnOftss5xY5JrEFJP9?usp=sharing
https://colab.research.google.com/drive/1YKXjCPglehTTKcoE82sUkh0feHmEjlQ9?usp=sharing
[https://www.kaggle.com/datasets/shantanugarg274/lung-cancer-prediction-dataset?resource=download]

#Lung Cancer Prediction using Machine Learning
##Introduction

Lung cancer remains one of the leading causes of cancer-related deaths worldwide, largely due to late diagnosis and the aggressive nature of the disease. Early detection through predictive modeling could significantly improve patient outcomes. In this project, we develop a predictive machine learning model to classify whether an individual has lung cancer (pulmonary disease) based on various risk factors and symptoms. Our objective is to compare several classification approaches – including a Random Forest, Support Vector Machine (SVM), Extreme Gradient Boosting (XGBoost), and a Multi-Layer Perceptron (MLP) neural network – to determine which yields the best performance in predicting lung cancer presence. We also analyze which features are most influential in the predictions.

##Summary of Findings:

Through a comprehensive exploratory data analysis and rigorous modeling, we found that ensemble tree methods (Random Forest and XGBoost) and SVM achieved high accuracy (around 89–90% on the test set) with excellent discrimination (Area Under the ROC Curve ~0.91–0.92). The Random Forest slightly outperformed others in accuracy. Key risk factors such as smoking status, certain respiratory symptoms (e.g., throat discomfort, breathing issues), and blood oxygen levels emerged as the most important predictors in the model. We also observed that the neural network (MLP) performed comparably but had a slightly higher false-negative rate, which is an important consideration in medical diagnosis. The following sections detail the data used, preprocessing steps, modeling approach, results, and implications for further improvement.

##Data Description

The dataset for this project was obtained from a Kaggle lung cancer risk factors dataset. It consists of 5,000 patient records and 18 columns (17 features and 1 target). Each record corresponds to one individual, with a binary target indicating Pulmonary Disease (“YES” for lung cancer presence, “NO” for absence). The features include a mix of demographic attributes, lifestyle factors, family history, and clinical symptoms or measurements related to lung health. Table 1 summarizes the features:
Age: Age of the individual in years (numeric).
Gender: Sex of the patient (1 = Male, 0 = Female).
Smoking: Whether the individual is a smoker (1 = Yes, 0 = No).
Finger Discoloration: Whether the patient has yellowing or discoloration of fingers (often from nicotine) – this can be a sign of heavy smoking.
Mental Stress: Presence of significant mental stress or anxiety (1 = Yes).
Exposure to Pollution: High exposure to air pollution (1 = Yes). (The dataset documentation defines this as a binary indicator of exposure to polluted environments.)
Long Term Illness: Presence of any long-term chronic disease (1 = Yes).
Energy Level: A quantitative measure of the patient’s energy or fatigue level. (Higher might indicate more energy; in data, this ranged roughly 0–100.)
Immune Weakness: Whether the patient has a weakened immune system (1 = Yes).
Breathing Issue: Presence of breathing difficulties (e.g., dyspnea or wheezing) (1 = Yes).
Alcohol Consumption: Whether the patient is a regular alcohol consumer (1 = Yes).
Throat Discomfort: Presence of throat discomfort (such as persistent cough or difficulty swallowing) (1 = Yes).
Oxygen Saturation: Blood oxygen saturation level (%) – a numeric feature typically around 90–100 in healthy individuals.
Chest Tightness: Experience of chest tightness or pain (1 = Yes).
Family History: Family history of lung cancer (genetic predisposition) (1 = Yes).
Smoking Family History: Whether there is a family history of smoking (e.g., parents or close family are/were smokers – indicates second-hand smoke exposure) (1 = Yes).
Stress Immune: A derived measure relating stress to immune response (numeric or categorical, encoded as numeric). (In the data, this was encoded as a numeric feature; it may represent combined effects of stress on immunity.)
Pulmonary Disease: Target variable – whether the patient has lung cancer (YES or NO).

There were no missing values in the provided CSV. All features were either numeric or already encoded as binary indicators, so we did not need to perform one-hot encoding. We verified class balance: approximately 2037 positive cases vs 2963 negative cases (~40.7% prevalence of “YES”), indicating a moderate class imbalance. The data was randomly split into training and testing sets (80/20 stratified split, detailed in the next section). Before modeling, continuous features like age, energy level, and oxygen saturation were standardized (mean=0, std=1) for models sensitive to feature scaling (SVM and MLP), while binary features were left as 0/1. Figure 1 below shows the distribution of the target classes in the dataset, confirming that both classes are well-represented (though “NO” is about 1.5 times more frequent than “YES”):

![image](https://github.com/user-attachments/assets/01cd073c-0355-4f2f-a4e0-c637c51f7311)



