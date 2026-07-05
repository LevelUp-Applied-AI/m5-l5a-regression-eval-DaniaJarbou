"""
Module 5 Week A — Lab: Regression & Evaluation

Build and evaluate logistic and linear regression models on the
Petra Telecom customer churn dataset.

Run: python lab_regression.py
"""

import pandas as pd
import numpy as np
import os
from sklearn.model_selection import train_test_split, cross_val_score, StratifiedKFold
from sklearn.linear_model import LogisticRegression, Ridge, Lasso
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.metrics import (classification_report,  ConfusionMatrixDisplay,
                             mean_absolute_error, r2_score )
import matplotlib.pyplot as plt

def load_data(filepath="data/telecom_churn.csv"):
    """Load the telecom churn dataset.

    Returns:
        DataFrame with all columns.
    """
   
      
    possible_path=[
        filepath,
        os.path.join("starter",filepath),
        os.path.join("starter","data","telecom_churn.csv"),
        os.path.join("data", "telecom_churn.csv"),
    ]
    
    for path in possible_path:
        if os.path.exists(path):
            filepath=path
            break
    df=pd.read_csv(filepath)    
    print(f"Dataset Shape: {df.shape}")
    print("\nMissing Values:\n", df.isnull().sum())
    print("\nTarget Distribution (churned):\n",
     df['churned'].value_counts(normalize=True))


    return df


def split_data(df, target_col, test_size=0.2, random_state=42 ,stratify= True):
    """Split data into train and test sets with stratification.

    Args:
        df: DataFrame with features and target.
        target_col: Name of the target column.
        test_size: Fraction for test set.
        random_state: Random seed.

    Returns:
        Tuple of (X_train, X_test, y_train, y_test).
    """
    #  Separate features and target, then split with stratification
    X = df.drop(columns=[target_col])
    y = df[target_col]
    
    #if target_col =="monthly_charges" can not use stratify
    if target_col == "monthly_charges":
        stratify_y = None
    else:
        stratify_y = y 


    X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=test_size, random_state=random_state, stratify=stratify_y
    )
    
    #Print sizes and churn rates
    print(f"Splitting data for target: {target_col}")
    print(f"Train size: {len(X_train)}, Test size: {len(X_test)}")

    
    if stratify_y is not None:
        print(f"Train churn rate: {y_train.mean():.2%}")
        print(f"Test churn rate: {y_test.mean():.2%}")
        
    return X_train, X_test, y_train, y_test




def build_logistic_pipeline():
    """Build a Pipeline with StandardScaler and LogisticRegression.

    Returns:
        sklearn Pipeline object.
    """
    #  Create and return a Pipeline with two steps


    #create piplime
    pip = Pipeline([
        ('scaler' , StandardScaler()),
       ( 'model',LogisticRegression(
        random_state=42, 
        max_iter=1000, 
        class_weight="balanced"
    ))
    ])
      
    return pip





def build_ridge_pipeline():
    """Build a Pipeline with StandardScaler and Ridge regression.

    Returns:
        sklearn Pipeline object.
    """
    #  Create and return a Pipeline for Ridge regression
    pip =Pipeline([
    ('scaler', StandardScaler()),
       ( 'model',Ridge(alpha=1.0))
    ])
    return pip

def evaluate_classifier(pipeline, X_train, X_test, y_train, y_test):
    """Train the pipeline and return classification metrics.

    Args:
        pipeline: sklearn Pipeline with a classifier.
        X_train, X_test: Feature arrays.
        y_train, y_test: Label arrays.

    Returns:
        Dictionary with keys: 'accuracy', 'precision', 'recall', 'f1'.
    """
    # Fit the pipeline on training data, predict on test, compute metrics
    #train
    pipeline.fit(X_train,y_train)
    #predict
    y_pred = pipeline.predict(X_test)

    ConfusionMatrixDisplay.from_predictions(y_test, y_pred)
    plt.title("Confusion Matrix - Logistic Regression")
    plt.show() 
    
    report = classification_report(y_test,y_pred,output_dict=True)

    return {
        'accuracy': report['accuracy'],
        'precision': report['weighted avg']['precision'],
        'recall': report['weighted avg']['recall'],
        'f1': report['weighted avg']['f1-score']
    }
    


def evaluate_regressor(pipeline, X_train, X_test, y_train, y_test):
    """Train the pipeline and return regression metrics.

    Args:
        pipeline: sklearn Pipeline with a regressor.
        X_train, X_test: Feature arrays.
        y_train, y_test: Target arrays.

    Returns:
        Dictionary with keys: 'mae', 'r2'.
    """
    #  Fit the pipeline, predict, and compute MAE and R²
    pipeline.fit(X_train, y_train)
    y_pred = pipeline.predict(X_test)

    #metrics 
    mae= mean_absolute_error(y_test,y_pred)
    r2= r2_score(y_test,y_pred)
    #results
    result = {
        'mae':mae,
        'r2':r2
    }
    return result


def run_cross_validation(pipeline, X_train, y_train, cv=5):
    """Run stratified cross-validation on the pipeline.

    Args:
        pipeline: sklearn Pipeline.
        X_train: Training features.
        y_train: Training labels.
        cv: Number of folds.

    Returns:
        Array of cross-validation scores.
    """
    #  Run cross_val_score with StratifiedKFold
    cv_split = StratifiedKFold(n_splits=cv, shuffle=True, random_state=42)

    scores= cross_val_score(pipeline,X_train,y_train,cv=cv_split,scoring="accuracy")  

    return scores

def build_lasso_pipeline ():
    #Build a Pipeline with StandardScaler and Lasso regression.

    pip=Pipeline([
       ( 'scaler', StandardScaler()),
       ('model',Lasso(alpha=0.1))
    ])
   

    return pip

def compare_coefficients(ridge_pipe, lasso_pipe, feature_names):
    #print Ridge vs Lasso coefficients side by side.
     ridge_coefs = ridge_pipe.named_steps['model'].coef_
     lasso_coefs = lasso_pipe.named_steps['model'].coef_
     print("\nRidge vs Lasso Coefficients Comparison\n")
     print(f"{'Feature':<20} {'Ridge':>10} {'Lasso':>10}")
     print("-" * 45)

     for name, r, l in zip(feature_names, ridge_coefs, lasso_coefs):
        print(f"{name:<20} {r:>10.4f} {l:>10.4f}")

if __name__ == "__main__":
    df = load_data()
    if df is not None:
        print(f"Loaded {len(df)} rows, {df.shape[1]} columns")

        # Select numeric features for classification
        numeric_features = ["tenure", "monthly_charges", "total_charges",
                           "num_support_calls", "senior_citizen",
                           "has_partner", "has_dependents"]

        # Classification: predict churn
        df_cls = df[numeric_features + ["churned"]].dropna()
        split = split_data(df_cls, "churned")
        if split:
            X_train, X_test, y_train, y_test = split
            pipe = build_logistic_pipeline()
            if pipe:
                metrics = evaluate_classifier(pipe, X_train, X_test, y_train, y_test)
                print(f"Logistic Regression: {metrics}")

                scores = run_cross_validation(pipe, X_train, y_train)
                if scores is not None:
                    print(f"CV: {scores.mean():.3f} +/- {scores.std():.3f}")

        # Regression: predict monthly_charges
        df_reg = df[["tenure", "total_charges", "num_support_calls",
                     "senior_citizen", "has_partner", "has_dependents",
                     "monthly_charges"]].dropna()
        split_reg = split_data(df_reg, "monthly_charges")
        if split_reg:
            X_tr, X_te, y_tr, y_te = split_reg
            ridge_pipe = build_ridge_pipeline()
            if ridge_pipe:
                reg_metrics = evaluate_regressor(ridge_pipe, X_tr, X_te, y_tr, y_te)
                print(f"Ridge Regression: {reg_metrics}")
                
                lasso_pipe = build_lasso_pipeline()
                lasso_pipe.fit(X_tr, y_tr)

                compare_coefficients(ridge_pipe, lasso_pipe, X_tr.columns)




#Task 7 Summary 
'''
1. Important Features:
The results show that 'tenure' and 'total_charges' are the strongest predictors of churn. These features have the largest coefficients in both Ridge and Lasso models.

2. Model Performance (Churn):
The model achieved an accuracy of approximately 63% and a recall of about 63%. Since the dataset is imbalanced, recall is important because it measures how well the model detects customers who actually churn.

3. Lasso Observations:
Lasso reduced the magnitude of several feature coefficients but did not eliminate any of them completely. This indicates that most features still contribute slightly to the prediction.

4. Recommendations:
More advanced models such as Random Forest or Gradient Boosting could improve performance, especially recall, and better capture non-linear relationships in the data.
'''