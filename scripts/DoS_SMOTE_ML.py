import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.linear_model import LogisticRegression
from xgboost import XGBClassifier, plot_importance
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, make_scorer
from databalancing import prepare_norm_balanced_data, apply_smote
from sklearn.metrics import confusion_matrix
from sklearn.model_selection import learning_curve
import matplotlib.pyplot as plt
from sklearn.model_selection import cross_val_score
from scipy.stats import wilcoxon
import itertools
from sklearn.model_selection import cross_validate

def wilcoxon_signed_rank_test(cv_scores, pair_items=2):
    # Step 1: Generate all unique pairs of models for comparison
    model_pairs = list(itertools.combinations(cv_scores.keys(), pair_items))

    # Step 2: Perform Wilcoxon signed-rank tests for each pair
    for model1, model2 in model_pairs:
        scores1 = cv_scores[model1]
        scores2 = cv_scores[model2]
        stat, p = wilcoxon(scores1, scores2)
        
        # Step 3: Interpret the results
        print(f"\nWilcoxon test between {model1} and {model2}: Statistics={stat}, p-value={p}")
        alpha = 0.05
        if p > alpha:
            print(f'No significant difference between {model1} and {model2} models (fail to reject H0)')
        else:
            print(f'Significant difference between {model1} and {model2} models (reject H0)')

def k_cross_validation(models, x_train, y_train, cv=5):
    # Scoring functions to use
    scoring = {'accuracy': make_scorer(accuracy_score),
           'precision_macro': make_scorer(precision_score, average='macro'),
           'recall_macro': make_scorer(recall_score, average='macro'),
           'f1_macro': make_scorer(f1_score, average='macro')}
    
    # Evaluate each model using cross-validation and display detailed metrics
    for name, model in models.items():
        cv_results = cross_validate(model, x_train, y_train, cv=cv, scoring=scoring)
        print(f"{name} - Cross-validated accuracy scores: {cv_results['test_accuracy']}")
        print(f"{name} - Cross-validated precision_macro scores: {cv_results['test_precision_macro']}")
        print(f"{name} - Cross-validated recall_macro scores: {cv_results['test_recall_macro']}")
        print(f"{name} - Cross-validated f1_macro scores: {cv_results['test_f1_macro']}")

def top_features_xgboost(x_train_norm, y_train, num_features=20):
    # Feature importance using XGBoost only on the original data (Preprocessed)
    xgb_model = XGBClassifier(use_label_encoder=False, eval_metric='mlogloss')
    xgb_model.fit(x_train_norm, y_train)

    # Get the feature scores as a dictionary
    f_score = xgb_model.get_booster().get_score(importance_type='weight')
    sorted_f_score = sorted(f_score.items(), key=lambda kv: kv[1], reverse=True)[:num_features]
    top_features = [feature for feature, _ in sorted_f_score]
    print(top_features)

    # Plot feature importances
    plt.figure(figsize=(10, 15))  # Adjust the size as needed
    plot_importance(xgb_model, height=0.8, ax=plt.gca(), importance_type='weight', max_num_features=num_features)

    # Save the plot to a file with high resolution
    plt.savefig("plots/top_feature_importances.png", dpi=300, bbox_inches='tight')

def smote_training(models, x_train_norm, y_train):
    # Benign Ratio Ranges
    benign_ratios = [0.5, 0.7, 0.8]

    # Loop through each benign ratio
    for benign_ratio in benign_ratios:
        print(f"\nApplying SMOTE with benign ratio: {benign_ratio}")
        x_resampled, y_resampled = apply_smote(x_train_norm, y_train, benign_ratio=benign_ratio)

        # Evaluate each model using cross-validation and display detailed metrics
        k_cross_validation(models, x_resampled, y_resampled, cv=5)
    
if __name__ == "__main__":
    top_features = [
        'Init Fwd Win Byts', 'Dst Port', 'Fwd Seg Size Min', 'Fwd IAT Min', 
        'Flow IAT Min', 'Flow Duration', 'FIN Flag Cnt', 'Fwd Header Len', 
        'Bwd Pkt Len Std', 'Flow IAT Mean', 'Fwd Pkts/s', 'TotLen Bwd Pkts', 
        'Init Bwd Win Byts', 'Bwd IAT Mean', 'TotLen Fwd Pkts', 'Fwd Pkt Len Max', 
        'Bwd Pkts/s', 'Flow IAT Max', 'Flow Byts/s', 'Bwd Pkt Len Mean'
    ]

    df = pd.read_csv('data/cleaned_combined.csv')
    x_train_norm, x_test_norm, y_train, y_test, label_map = prepare_norm_balanced_data(df, top_features)

    # Initialize models
    models = {
        'Random Forest': RandomForestClassifier(random_state=42),
        'Decision Tree': DecisionTreeClassifier(random_state=42),
        'Logistic Regression': LogisticRegression(max_iter=1000),
        'XGBoost': XGBClassifier(use_label_encoder=False, eval_metric='mlogloss')
    }

    # Cross-validate the models using the original data
    k_cross_validation(models, x_train_norm, y_train, cv=5)

