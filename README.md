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
Exposure to Pollution: High exposure to air pollution (1 = Yes). 
Long Term Illness: Presence of any long-term chronic disease (1 = Yes).
Energy Level: A quantitative measure of the patient’s energy or fatigue level. (Higher might indicate more energy; in data, this ranged roughly 0–100.)
Immune Weakness: Whether the patient has a weakened immune system (1 = Yes).
Breathing Issue: Presence of breathing difficulties (e.g., dyspnea or wheezing) (1 = Yes).
Alcohol Consumption: Whether the patient is a regular alcohol consumer (1 = Yes).
Throat Discomfort: Presence of throat discomfort (1 = Yes).
Oxygen Saturation: Blood oxygen saturation level (%) – a numeric feature typically around 90–100 in healthy individuals.
Chest Tightness: Experience of chest tightness or pain (1 = Yes).
Family History: Family history of lung cancer (genetic predisposition) (1 = Yes).
Smoking Family History: Whether there is a family history of smoking (e.g., parents or close family are/were smokers – indicates second-hand smoke exposure) (1 = Yes).
Stress Immune: A derived measure relating stress to immune response. 
Pulmonary Disease: Target variable – whether the patient has lung cancer (YES or NO).

There were no missing values in the provided CSV. All features were either numeric or already encoded as binary indicators, so we did not need to perform one-hot encoding. We verified class balance: approximately 2037 positive cases vs 2963 negative cases (~40.7% prevalence of “YES”), indicating a moderate class imbalance. The data was randomly split into training and testing sets (80/20 stratified split). Before modeling, continuous features like age, energy level, and oxygen saturation were standardized (mean=0, std=1) for models sensitive to feature scaling (SVM and MLP), while binary features were left as 0/1. Figure 1 below shows the distribution of the target classes in the dataset, confirming that both classes are well-represented (though “NO” is about 1.5 times more frequent than “YES”):

![image](https://github.com/user-attachments/assets/01cd073c-0355-4f2f-a4e0-c637c51f7311)

Before modeling, we performed exploratory data analysis (EDA) to understand feature distributions and correlations. Notably, we observed that certain risk factors are much more prevalent in the positive cases than in the negatives. For example, smoking is prevalent among lung cancer patients in the data (about 93% of patients with cancer are smokers, compared to ~49% of the non-cancer group being smokers). Similarly, symptoms like breathing issues and throat discomfort are reported by a majority of cancer patients but only a small fraction of those without cancer. Figure 2 illustrates some of these differences for key features (smoking, smoking family history, throat discomfort, and breathing issue) between the “YES” and “NO” groups:

![image](https://github.com/user-attachments/assets/e6d9a4db-b259-4319-a2ae-ea9ec9aaa91f)

Figure 3 is a heatmap of Pearson correlation coefficients between all pairwise features. As expected, most features are relatively independent. Notable correlations include a moderate positive correlation (~0.46) between the target and Smoking and similarly between the target and Smoking Family History. There are also correlations among some features themselves – for instance, Finger Discoloration correlates with Smoking, and Breathing Issue correlates with Throat Discomfort and Chest Tightness. Overall, the multicollinearity is not severe, but the cluster of symptom variables shows some intercorrelation, and the cluster of smoking-related variables (smoking and family smoking history) is correlated:

![image](https://github.com/user-attachments/assets/7081f22c-b637-485f-8743-c51f11ac77fb)

#Models and Methods

To address the prediction task, we formulated it as a binary classification problem. We experimented with four different modeling techniques:
Random Forest (RF) Classifier: an ensemble of decision trees using bootstrap aggregation (bagging) and random feature selection for each split. RF can handle non-linear feature interactions well and provides feature importance estimates.

Support Vector Machine (SVM): We used an SVM with a Radial Basis Function (RBF) kernel, which can capture non-linear decision boundaries. The SVM finds an optimal hyperplane in a transformed feature space that maximizes the margin between the two classes.

Extreme Gradient Boosting (XGBoost): a gradient-boosted decision tree model. XGBoost builds an ensemble of trees in sequence, where each new tree corrects errors of the previous ones. It often achieves high accuracy by optimizing a carefully regularized objective.

Multi-Layer Perceptron (MLP): a feed-forward artificial neural network with one hidden layer (we used 2 hidden layers with 32 and 16 neurons, respectively). The MLP can learn complex non-linear relationships. Ours uses a sigmoid activation for outputs and was trained with a maximum of 500 iterations.

Training Procedure: We split the data into a training set and test set, maintaining the 60/40 class ratio in both. All model development was done on the training set, and the final results were evaluated on the hold-out test set to assess generalization performance. For SVM and MLP, input features were standardized (z-score scaling) because these models are sensitive to feature scale. We did not perform an extensive hyperparameter search due to time constraints; instead, we used reasonable defaults or common settings for each model: for example, the Random Forest used 100 trees with default depth, the SVM used the RBF kernel with default regularization parameter C, XGBoost was run with its default tree depth and learning rate (with eval_metric='logloss'), and the MLP had a relatively small architecture as noted above to mitigate overfitting. Early stopping or cross-validation could be employed in future work to further tune these hyperparameters.

#Results and Interpretation

The Random Forest achieved the highest accuracy (~90.4%), correctly classifying 904 out of 1000 test instances. Its ROC AUC was 0.92, indicating excellent discrimination ability (it can separate positive vs negative cases very well). The SVM’s performance was a close second, with virtually the same AUC (0.924) and a slightly lower accuracy (88.9%). XGBoost also performed comparably (89.0% accuracy, AUC ~0.91). The MLP neural network trailed slightly, with about 87.4% accuracy and AUC ~0.90. To put these numbers in context, an accuracy around 90% and AUC > 0.90 is quite high for a medical classification task. This suggests the features in this dataset carry a lot of signal for lung cancer risk. The high AUC values mean that, if we were to rank patients by the model’s predicted risk score, a patient with cancer is usually ranked higher than a patient without cancer, ~90+% of the time – a very desirable property for screening tools


In summary, Random Forest emerged as the best model in our evaluation, edging out SVM by a small margin in accuracy while maintaining an equally high AUC and slightly fewer false negatives. The SVM and XGBoost are not far behind, and all three could be considered robust classifiers for this task. The neural network’s performance, while slightly lower, is still respectable given its simple architecture and limited tuning, with further optimization, it might improve. However, the tree-based models have the added advantage of interpretability via feature importance, which we explore next.

According to the Random Forest, the Smoking feature was the most important predictor by a substantial margin (importance ~0.212). This aligns with domain knowledge that smoking is the leading risk factor for lung cancer. The second most important was Energy Level (~0.155) – in the dataset, many cancer patients had markedly lower energy levels, which could reflect cancer-related fatigue. Next were Throat Discomfort (~0.111) and Breathing Issue (~0.107), both of which are symptoms often associated with lung tumors or respiratory distress. Oxygen Saturation (0.096) was also a strong predictor (patients with lung cancer tended to have slightly lower O2 saturation, potentially due to compromised lung function). Age ranked next (~0.081), which makes sense as risk increases with age. Interestingly, Smoking Family History had some importance (~0.048) – possibly capturing second-hand smoke exposure or genetic predisposition – though it was less important than the person’s smoking status. Features like Stress_Immune, Exposure to Pollution, and others had smaller contributions (all <0.04). It’s worth noting that some features may be correlated, so importance can be distributed among them. But the clear takeaway is that smoking and respiratory symptoms drive most of the predictive power in this dataset, which is consistent with medical expectations.


#Conclusion and Next Steps

In this study, we successfully developed and evaluated several machine learning models to predict lung cancer presence from patient data. All models achieved high accuracy and AUC, with the Random Forest performing best overall on our dataset (90% accuracy, AUC ~0.92). We found that smoking status and respiratory-related symptoms are the most critical factors associated with lung cancer risk in this data, which reinforces existing medical knowledge about risk factors. 

The SVM and XGBoost models also provided strong performance, suggesting that a non-linear decision boundary (SVM) and boosted trees are effective approaches for this problem. The neural network, while slightly behind, showed promise and could potentially improve with more tuning or more data. Limitations: Our current models were trained and evaluated on the same dataset, which, while split into train/test, comes from a single source. There is a risk of overfitting to the nuances of this specific dataset. The high accuracy might not fully generalize to new data from different populations or hospitals without further validation. Also, the dataset’s moderate imbalance means accuracy alone can be a bit misleading – hence our focus on AUC and examining false negatives explicitly. In a medical application, one might favor a model that sacrifices a bit of specificity to gain higher sensitivity. That could be achieved by adjusting the classification threshold or using cost-sensitive training. 


Next Steps and Improvements:
Hyperparameter Tuning: We can perform grid search or randomized search for each model to potentially improve performance. For example, tuning the number of trees or depth in Random Forest/XGBoost, or the C and gamma in SVM, or using a different architecture or learning rate for the MLP. This may yield incremental gains in accuracy or AUC.

Ensemble Approach: Given that the top three models all did well, an ensemble could be tried to see if a combination of models yields even better performance. Sometimes ensembles of diverse classifiers can reduce generalization error.

Focus on Recall: In practice, missing a lung cancer case is far more costly than a false alarm. We could adjust the decision threshold of our Random Forest (or others) to increase sensitivity. For instance, operating at a point on the ROC curve that gives, say, 95% sensitivity might increase false positives but ensure most true cases are caught. This could be complemented with medical follow-up tests to confirm.


In conclusion, the project demonstrates a successful application of machine learning for lung cancer risk prediction using patient survey data. The models can identify high-risk individuals with substantial accuracy, which could assist in early warning or screening programs. By focusing on interpretable features like smoking and symptoms, the model’s findings align with known medical risk factors, adding trust to its predictions. With further refinement and validation, such a model could be integrated into a clinical decision support system to flag patients who may benefit from early diagnostic screenings for lung cancer. Future work will aim to broaden the model’s scope and ensure its reliability and fairness before real-world deployment.

