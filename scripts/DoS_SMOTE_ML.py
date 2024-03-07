import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.linear_model import LogisticRegression
import xgboost as xgb
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
import time
from sklearn.metrics import log_loss
import os
from collections import Counter

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

def k_cross_validation(models, x_train, y_train, cv=5, benign_ratio=0.0):
    scoring = {
        'accuracy': make_scorer(accuracy_score),
        'precision_macro': make_scorer(precision_score, average='macro'),
        'recall_macro': make_scorer(recall_score, average='macro'),
        'f1_macro': make_scorer(f1_score, average='macro')
    }

    all_results = {}  # Store results for the current benign ratio

    for name, model in models.items():
        cv_results = cross_validate(model, x_train, y_train, cv=cv, scoring=scoring, return_train_score=True)
        cv_results['benign_ratio'] = benign_ratio  
        all_results[name] = cv_results  

        # Save the results for this benign ratio/model combination
        results_df = pd.DataFrame(cv_results)
        os.makedirs('results/smote', exist_ok=True)
        results_df.to_csv(f"results/smote/{name}_benign_{benign_ratio}_cv_results.csv", index=False)

    return all_results 


def plot_model_learning_curve(model, name, x_train, y_train, x_test, y_test, cv, train_sizes):
    """
    Plots the learning curve for a given model. Handles special case for XGBoost.
    """
    plt.figure(figsize=(8, 6))
    plt.title(f"Learning Curve: {name}")
    plt.xlabel("Training examples")
    plt.ylabel("Score")

    if name == 'XGBoost':
        plot_xgboost_learning_curve(model, x_train, y_train, x_test, y_test)
    else:
        plot_standard_model_learning_curve(model, x_train, y_train, cv, train_sizes)

    save_plot(name)

def plot_xgboost_learning_curve(model, x_train, y_train, x_test, y_test, metric='mlogloss'):
    eval_set = [(x_train, y_train), (x_test, y_test)]
    model.fit(x_train, y_train, eval_set=eval_set)
    
    results = model.evals_result()
    print(results['validation_0'].keys())
    epochs = len(results['validation_0'][metric])  
    x_axis = range(0, epochs)

    plt.plot(x_axis, results['validation_0'][metric], label='Train')  
    plt.plot(x_axis, results['validation_1'][metric], label='Validation')  
    plt.legend()
    plt.ylabel('Log Loss')
    plt.title('XGBoost Log Loss')

def plot_standard_model_learning_curve(model, x_train, y_train, cv, train_sizes):
    train_sizes, train_scores, test_scores = learning_curve(
        model, x_train, y_train, cv=cv, n_jobs=-1, train_sizes=train_sizes)
    
    train_scores_mean = np.mean(train_scores, axis=1)
    train_scores_std = np.std(train_scores, axis=1)
    test_scores_mean = np.mean(test_scores, axis=1)
    test_scores_std = np.std(test_scores, axis=1)

    plt.grid()
    plt.fill_between(train_sizes, train_scores_mean - train_scores_std, 
                     train_scores_mean + train_scores_std, alpha=0.1, color="r")
    plt.fill_between(train_sizes, test_scores_mean - test_scores_std, 
                     test_scores_mean + test_scores_std, alpha=0.1, color="g")
    plt.plot(train_sizes, train_scores_mean, 'o-', color="r", label="Training score")
    plt.plot(train_sizes, test_scores_mean, 'o-', color="g", label="Cross-validation score")

    plt.legend(loc="best")

def save_plot(name):
    """
    Saves the plot to a directory specific to the model name.
    """
    model_dir = f"plots/learning_curves/{name.replace(' ', '_')}"
    os.makedirs(model_dir, exist_ok=True)
    
    plot_path = f"{model_dir}/{name.replace(' ', '_')}_learning_curve.png"
    plt.savefig(plot_path, dpi=300, bbox_inches='tight')
    plt.close()

def plot_learning_curve(models, x_train, y_train, x_test, y_test, cv=5, train_sizes=np.linspace(0.1, 1.0, 5)):
    for name, model in models.items():
        plot_model_learning_curve(model, name, x_train, y_train, x_test, y_test, cv, train_sizes)

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
        k_cross_validation(models, x_resampled, y_resampled, cv=5, benign_ratio=benign_ratio)

def label_distribution_csv(y_train, label_mapping):
    """ This function plots a bar chart of the label distribution.
    Args:
        y_train: pandas Series or numpy array, training labels
        label_mapping: dict, a mapping of encoded labels to original labels

    Returns:
        None
    """
    label_counts = Counter(y_train)

    labels = [label_mapping.get(label, label) for label in label_counts.keys()]
    data = {
        "Label ID": label_counts.keys(),
        "Description": labels,
        "Count": label_counts.values(),
        "Percentage": [f"{(count / sum(label_counts.values())) * 100:.2f}%" for count in label_counts.values()],
    }

    df_table = pd.DataFrame(data)
    df_table = df_table.sort_values(by="Label ID").reset_index(drop=True)
    total_row = pd.DataFrame({'Label ID': '-', 'Description': 'Total',
                            'Count': df_table['Count'].sum(), 'Percentage': '100%'}, index=[0])
    df_table = pd.concat([df_table, total_row], ignore_index=True)

    # Save to CSV
    df_table.to_csv("label_distribution_table.csv", index=False)

    # Save the figure
    os.makedirs('results/data_distribution', exist_ok=True)
    df_table.to_csv('results/data_distribution/label_distribution.csv', index=False)

def save_original_raw_data(df):
    """ This function saves the original raw data to a CSV file.
    Args:
        df: pandas DataFrame, the original raw data

    Returns:
        None
    """
    label_counter = Counter(df['Label'])

    # Prepare data for saving
    label_counts = pd.DataFrame.from_dict(label_counter, orient='index', columns=['Count'])
    label_counts.index.name = 'Label'
    label_counts.reset_index(inplace=True)
    label_counts['Percentage'] = (label_counts['Count'] / label_counts['Count'].sum()) * 100

    # Create the output directory if it doesn't exist
    os.makedirs('results/data_distribution', exist_ok=True)

    # Save the label distribution to a CSV file
    label_counts.to_csv('results/data_distribution/raw_label_distribution.csv', index=False) 

    
if __name__ == "__main__":
    top_features = [
        'Init Fwd Win Byts', 'Dst Port', 'Fwd Seg Size Min', 'Fwd IAT Min', 
        'Flow IAT Min', 'Flow Duration', 'FIN Flag Cnt', 'Fwd Header Len', 
        'Bwd Pkt Len Std', 'Flow IAT Mean', 'Fwd Pkts/s', 'TotLen Bwd Pkts', 
        'Init Bwd Win Byts', 'Bwd IAT Mean', 'TotLen Fwd Pkts', 'Fwd Pkt Len Max', 
        'Bwd Pkts/s', 'Flow IAT Max', 'Flow Byts/s', 'Bwd Pkt Len Mean'
    ]

    df = pd.read_csv('data/cleaned_combined.csv')

    x_train_norm, x_test_norm, y_train, y_test, label_map = prepare_norm_balanced_data(df, top_features, remove_duplicates=True)

    # Initialize models
    models = {
        'Random Forest': RandomForestClassifier(random_state=42),
        'Decision Tree': DecisionTreeClassifier(random_state=42),
        'Logistic Regression': LogisticRegression(max_iter=1000),
        'XGBoost': XGBClassifier(use_label_encoder=False, eval_metric='mlogloss', n_estimators=100)
    }

    # Cross-validate the models using the original data
    # k_cross_validation(models, x_train_norm, y_train, cv=5)

    # Apply SMOTE to the training data
    smote_training(models, x_train_norm, y_train)

    # Plot learning curves for each model
    # plot_learning_curve(models, x_train_norm, y_train, x_test_norm, y_test, cv=5)
    