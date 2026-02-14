# Beta Bank Customer Churn Prediction Project

### Problem Context:
Development of a machine learning classification system to predict customer churn for Beta Bank, enabling proactive customer retention strategies by identifying clients likely to leave the bank based on their behavioral and demographic characteristics.

### Objective:
Build and optimize classification models to predict customer churn with a minimum F1-score of 0.59, using customer data including demographics, account information, and banking behavior patterns to enable targeted retention campaigns.

Technical Competencies Used:
Exploratory Data Analysis (EDA):
- Investigation of customer dataset with 10,000 records across 14 features
- Analysis of missing values in the Tenure column (909 missing values)
- Data quality assessment including duplicate detection
- Feature engineering for categorical variables

Data Preprocessing:
- Missing value treatment (Tenure column filled with zeros for logical business interpretation)
- One-hot encoding for categorical variables (Geography, Gender)
- Feature selection removing non-predictive columns (RowNumber, CustomerId, Surname)
- Data type conversions and standardization

Machine Learning Model Development:
- Logistic Regression: Baseline model with poor performance on imbalanced data
- Decision Tree Classifier: Moderate performance with hyperparameter tuning (max_depth optimization)
- Random Forest Classifier: Best performing model with comprehensive hyperparameter optimization

Class Imbalance Handling:
- Class weighting: Using class_weight='balanced' parameter
- Upsampling technique: Manual oversampling of minority class (10x replication)
- Comparative analysis of different balancing approaches

Model Optimization:
- Hyperparameter tuning for Random Forest (n_estimators, max_depth, min_samples_split)
- Threshold optimization for precision-recall balance
- Systematic evaluation across parameter combinations

Model Evaluation:
- Multiple metrics: Precision, Recall, F1-score
- Confusion matrix analysis
- ROC-AUC curve analysis (0.86 AUC score)
- Proper train-validation-test split methodology

### Main Libraries:
- pandas: Data manipulation and preprocessing
- numpy: Numerical operations and array handling
- scikit-learn: Machine learning algorithms and evaluation
  - RandomForestClassifier, DecisionTreeClassifier, LogisticRegression
  - train_test_split, evaluation metrics
  - StandardScaler for preprocessing
- matplotlib: Data visualization and ROC curve plotting

### Final Results:
Successful development of a high-performance churn prediction system:

Model Performance:
- Final F1-score: 0.6424 on test set (exceeding 0.59 minimum requirement)
- ROC-AUC: 0.86 indicating excellent discriminative ability
- Optimal threshold: 0.45 for balanced precision-recall trade-off

Best Model Configuration:
- Random Forest Classifier with class balancing
- Hyperparameters: n_estimators=50, max_depth=10, min_samples_split=5
- Threshold optimization: 0.45 for optimal F1-score

Business Impact:
- Enables proactive identification of at-risk customers
- Supports targeted retention campaigns with 64% accuracy in identifying churners
- Provides interpretable model for business decision-making
- ROC-AUC of 0.86 demonstrates strong predictive capability across all threshold levels

Technical Achievements:
- Proper handling of class imbalance through multiple techniques
- Comprehensive hyperparameter optimization methodology
- Robust evaluation using separate test set
- Threshold tuning for business-specific precision-recall requirements

The project demonstrates proficiency in end-to-end machine learning pipeline development, from data preprocessing through model optimization to business-ready deployment for customer churn prediction.
