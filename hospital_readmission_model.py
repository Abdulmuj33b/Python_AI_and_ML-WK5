# # Import required libraries
# import pandas as pd  # Data manipulation and analysis
# import numpy as np   # Numerical operations
# from sklearn.preprocessing import RobustScaler  # Scaling that handles outliers
# from sklearn.model_selection import train_test_split  # Data splitting utilities
# import lightgbm as lgb  # Gradient Boosting framework optimized for speed
# import shap  # SHapley Additive exPlanations for model interpretation

# Load dataset from CSV file
# Assumption: Dataset contains patient records with '30_day_readmit' as target variable
data = pd.read_csv('ehr_data.csv')  # Replace with actual dataset path

# Define preprocessing pipeline as a function for modularity and reusability
def preprocess(df):
    """
    Comprehensive data preprocessing function
    Args:
        df: Raw DataFrame containing patient records
    Returns:
        Processed DataFrame ready for modeling
    """
    
    # 1. Handle Missing Values
    # For medication adherence, assume missing = no adherence (0)
    df['medication_adherence'].fillna(0, inplace=True)
    # For lab values like A1C, fill missing with median (robust to outliers)
    df['last_a1c'].fillna(df['last_a1c'].median(), inplace=True)
    
    # 2. Feature Engineering - Create new predictive features
    # Comorbidity index: Count of chronic conditions (hypertension, CKD, CAD)
    df['comorbidity_index'] = df[['hypertension', 'ckd', 'cad']].sum(axis=1)
    # Prior admission risk flag: High risk if >3 previous admissions
    df['prior_admit_high_risk'] = np.where(df['prior_admissions'] > 3, 1, 0)
    
    # 3. Encode Categorical Variables
    # Convert insurance type to one-hot encoded columns
    # Creates separate binary columns for each insurance category
    df = pd.get_dummies(df, columns=['insurance_type'])
    
    # 4. Normalize Numerical Features
    # Initialize scaler that uses median/IQR to minimize outlier impact
    scaler = RobustScaler()
    # Select numerical columns to scale
    num_cols = ['age', 'bmi', 'last_a1c']
    # Apply scaling and replace original columns
    df[num_cols] = scaler.fit_transform(df[num_cols])
    
    return df

# Execute preprocessing pipeline
processed_data = preprocess(data)

# Prepare feature matrix (X) and target vector (y)
# Drop target column to create feature matrix
X = processed_data.drop('30_day_readmit', axis=1)
# Extract target variable column
y = processed_data['30_day_readmit']

# Data Splitting Strategy: 60% train, 20% validation, 20% test
# First split: Separate 60% training data from remaining 40%
X_train, X_temp, y_train, y_temp = train_test_split(
    X, y, 
    test_size=0.4,  # Reserve 40% for validation + test
    stratify=y,      # Maintain class distribution ratios
    random_state=42  # Seed for reproducibility
)

# Second split: Divide temporary set equally into validation and test
X_val, X_test, y_val, y_test = train_test_split(
    X_temp, y_temp, 
    test_size=0.5,   # Split temp set 50/50 (20% total each)
    stratify=y_temp, # Maintain class distribution
    random_state=42  # Consistent seed
)

# Verify split sizes and proportions
total_samples = len(X)
print(f"Total samples: {total_samples}")
print(f"Train size: {len(X_train)} ({len(X_train)/total_samples*100:.1f}%)")
print(f"Validation size: {len(X_val)} ({len(X_val)/total_samples*100:.1f}%)")
print(f"Test size: {len(X_test)} ({len(X_test)/total_samples*100:.1f}%)")
print(f"Class distribution in train set: {y_train.value_counts(normalize=True)}")

# Initialize LightGBM Classifier with optimized settings
model = lgb.LGBMClassifier(
    boosting_type='goss',  # Gradient-based One-Side Sampling (handles imbalance)
    num_leaves=31,         # Complexity control (default)
    learning_rate=0.05,    # Smaller rate = better convergence, more trees
    n_estimators=1000,     # Generous number (early stopping will determine actual)
    reg_alpha=0.2,         # L1 regularization to prevent overfitting
    reg_lambda=0.4,        # L2 regularization for smoother learning
    random_state=42,       # Reproducible results
    is_unbalance=True,     # Adjusts for class imbalance automatically
    metric='binary_logloss' # Optimization metric
)

# Train model with validation-based early stopping
model.fit(
    X_train, y_train,          # Training data
    eval_set=[(X_val, y_val)],  # Validation set for monitoring
    eval_metric='logloss',      # Evaluation metric (cross-entropy)
    early_stopping_rounds=50,   # Stop if no improvement for 50 rounds
    verbose=10                 # Print evaluation every 10 iterations
)

# Generate SHAP values for model interpretability
# Initialize TreeExplainer with trained model
explainer = shap.TreeExplainer(model)
# Calculate SHAP values for test set
shap_values = explainer.shap_values(X_test)
# Generate summary plot of feature importance
shap.summary_plot(shap_values, X_test, feature_names=X.columns.tolist())

# Generate feature importance plot for clinical insights
lgb.plot_importance(model, max_num_features=15, importance_type='gain')

# Model Evaluation on unseen test set
test_preds = model.predict(X_test)          # Class predictions
test_probs = model.predict_proba(X_test)[:, 1]  # Probability scores

# Generate evaluation metrics (placeholder - implement actual metrics)
from sklearn.metrics import classification_report
print("\nTest Set Performance:")
print(classification_report(y_test, test_preds))

# Save model in LightGBM's native format for efficient deployment
model.booster_.save_model('readmission_model.txt')

# Additional recommended steps:
# 1. Save preprocessing artifacts (scaler, imputation values)
# 2. Implement full evaluation metrics (AUC, precision-recall)
# 3. Create calibration plot for probability validation
# 4. Generate error analysis by patient subgroups
