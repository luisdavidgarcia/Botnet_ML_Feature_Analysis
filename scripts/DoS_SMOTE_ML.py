import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.linear_model import LogisticRegression
from xgboost import XGBClassifier
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, make_scorer
from databalancing import prepare_norm_balanced_data, apply_smote
from sklearn.metrics import confusion_matrix
from sklearn.model_selection import learning_curve
import matplotlib.pyplot as plt
from sklearn.model_selection import cross_val_score
from scipy.stats import wilcoxon
import itertools
from sklearn.model_selection import cross_validate

# def plot_learning_curves(model, X, y, model_name, ratio):
#     train_sizes, train_scores, test_scores = learning_curve(model, X, y, n_jobs=-1, cv=5, train_sizes=np.linspace(.1, 1.0, 5), verbose=0, random_state=42)
#     train_scores_mean = np.mean(train_scores, axis=1)
#     test_scores_mean = np.mean(test_scores, axis=1)
    
#     plt.figure()
#     plt.xlabel("Training examples")
#     plt.ylabel("Score")
#     plt.plot(train_sizes, train_scores_mean, 'o-', color="r", label="Training score")
#     plt.plot(train_sizes, test_scores_mean, 'o-', color="g", label="Cross-validation score")
#     plt.title(f"Learning Curve: {model_name}, Ratio {ratio}")
#     plt.legend(loc="best")
#     plt.grid()
    
#     # Save the plot to a file
#     plt.savefig(f"learning_curve_{model_name}_ratio_{ratio}.png")
#     plt.close()  # Close the plot to free memory


# # Load the dataset
# df = pd.read_csv('02-15-2018.csv')
# x_train_norm, x_test_norm, y_train, y_test, label_map = prepare_norm_balanced_data(df)

# # Benign Ratio Ranges
# benign_ratios = [0.5, 0.7, 0.8]

# # Store results
# results = []

# # Initialize a dictionary to store cross-validated scores
# cv_scores = {
#     'Random Forest': [],
#     'Decision Tree': [],
#     'Logistic Regression': [],
#     'XGBoost': []
# }

# # Train and evaluate models for each benign ratio using SMOTE (synthetic data)
# for benign_ratio in benign_ratios:
#     print("---------------------------------------------------------------------")
#     print(f"Applying SMOTE with benign ratio: {benign_ratio}")
#     x_resampled, y_resampled = apply_smote(x_train_norm, y_train, begnin_ratio=benign_ratio)

#     print(f"Len of x_train_norm: {len(x_train_norm)} and len of x_resampled: {len(x_resampled)}")
#     print("---------------------------------------------------------------------")

#     # Initialize models
#     rf_model = RandomForestClassifier(random_state=42)
#     dt_model = DecisionTreeClassifier(random_state=42)
#     lr_model = LogisticRegression(max_iter=1000)
#     xgb_model = XGBClassifier(use_label_encoder=False, eval_metric='mlogloss')

#     # Train and evaluate each model
#     model_list = [(rf_model, 'Random Forest'), (dt_model, 'Decision Tree'), 
#               (lr_model, 'Logistic Regression'), (xgb_model, 'XGBoost')]
#     scoring_metrics = ['accuracy', 'precision_macro', 'recall_macro', 'f1_macro']

#     for model, name in model_list:
#         # FIXME: Plot learning curves here for each model
#         # plot_learning_curves(model, x_resampled, y_resampled, name, benign_ratio)

#         # Instead of fitting here, calculate cross-validated scores
#         accuracy_scores = cross_val_score(model, x_resampled, y_resampled, cv=5, scoring='accuracy')
        
#         # Store the mean accuracy score
#         cv_scores[name].append(np.mean(accuracy_scores))

#         model.fit(x_resampled, y_resampled)
#         y_pred = model.predict(x_test_norm)

#         # Calculate metrics
#         accuracy = accuracy_score(y_test, y_pred)
#         precision = precision_score(y_test, y_pred, average='macro')
#         recall = recall_score(y_test, y_pred, average='macro')
#         f1 = f1_score(y_test, y_pred, average='macro')

#         print(f"{name} - Accuracy: {accuracy}, Precision: {precision}, Recall: {recall}, F1 Score: {f1}")

#         print(f"Results for {name}:")
#         for metric in scoring_metrics:
#             scores = cross_val_score(model, x_resampled, y_resampled, cv=5, scoring=metric)
#             print(f"Cross-validated {metric} scores: {scores}")
#             print(f"Mean {metric}: {np.mean(scores)}, Standard Deviation {metric}: {np.std(scores)}")
#         print("-------------------------------------------------")

#         # # Compute confusion matrix
#         # cm = confusion_matrix(y_test, y_pred)

#         # # For multiclass, calculate metrics per class
#         # FP = cm.sum(axis=0) - np.diag(cm)  
#         # FN = cm.sum(axis=1) - np.diag(cm)
#         # TP = np.diag(cm)
#         # TN = cm.sum() - (FP + FN + TP)

#         # # Compute FPR and FNR for each class and store the average if needed
#         # FPR = FP / (FP + TN)
#         # FNR = FN / (TP + FN)

#         # # Calculate average FPR and FNR across all classes
#         # avg_FPR = np.mean(FPR)
#         # avg_FNR = np.mean(FNR)

#         # # Store results including average FPR and FNR
#         # results.append({
#         #     "Benign Ratio": benign_ratio,
#         #     "Model": name,
#         #     "Average FPR": avg_FPR,
#         #     "Average FNR": avg_FNR
#         # })


# # results_df = pd.DataFrame(results)

# # # Save results to a CSV file
# # results_df.to_csv("smote_results.csv", index=False)

# # After evaluating all models, perform Wilcoxon signed-rank tests between models
# # Example: Comparing Random Forest and Decision Tree
# rf_accuracy = cv_scores['Random Forest']
# dt_accuracy = cv_scores['Decision Tree']

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

if __name__ == "__main__":
    # Load the dataset
    df = pd.read_csv('data/raw/02-15-2018.csv')
    x_train_norm, x_test_norm, y_train, y_test, label_map = prepare_norm_balanced_data(df)

    # Benign Ratio Ranges
    benign_ratios = [0.5, 0.7, 0.8]

    # Initialize a dictionary to store cross-validated scores
    cv_scores = {'Random Forest': [], 'Decision Tree': [], 'Logistic Regression': [], 'XGBoost': []}

    # Initialize models
    models = {
        'Random Forest': RandomForestClassifier(random_state=42),
        'Decision Tree': DecisionTreeClassifier(random_state=42),
        'Logistic Regression': LogisticRegression(max_iter=1000),
        'XGBoost': XGBClassifier(use_label_encoder=False, eval_metric='mlogloss')
    }

    # Loop through each benign ratio
    for benign_ratio in benign_ratios:
        print(f"\nApplying SMOTE with benign ratio: {benign_ratio}")
        x_resampled, y_resampled = apply_smote(x_train_norm, y_train, benign_ratio=benign_ratio)

