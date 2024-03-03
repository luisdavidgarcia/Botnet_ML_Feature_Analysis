import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.linear_model import LogisticRegression
from xgboost import XGBClassifier
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
from databalancing import prepare_norm_balanced_data, apply_smote
from sklearn.metrics import confusion_matrix
from sklearn.model_selection import learning_curve
import matplotlib.pyplot as plt

def plot_learning_curves(model, X, y, model_name, ratio):
    train_sizes, train_scores, test_scores = learning_curve(model, X, y, n_jobs=-1, cv=5, train_sizes=np.linspace(.1, 1.0, 5), verbose=0, random_state=42)
    train_scores_mean = np.mean(train_scores, axis=1)
    test_scores_mean = np.mean(test_scores, axis=1)
    
    plt.figure()
    plt.xlabel("Training examples")
    plt.ylabel("Score")
    plt.plot(train_sizes, train_scores_mean, 'o-', color="r", label="Training score")
    plt.plot(train_sizes, test_scores_mean, 'o-', color="g", label="Cross-validation score")
    plt.title(f"Learning Curve: {model_name}, Ratio {ratio}")
    plt.legend(loc="best")
    plt.grid()
    
    # Save the plot to a file
    plt.savefig(f"learning_curve_{model_name}_ratio_{ratio}.png")
    plt.close()  # Close the plot to free memory


# Load the dataset
df = pd.read_csv('02-15-2018.csv')
x_train_norm, x_test_norm, y_train, y_test, label_map = prepare_norm_balanced_data(df)

# Benign Ratio Ranges
benign_ratios = [0.5, 0.7, 0.8]

# Store results
results = []

# Train and evaluate models for each benign ratio using SMOTE (synthetic data)
for benign_ratio in benign_ratios:
    print("---------------------------------------------------------------------")
    print(f"Applying SMOTE with benign ratio: {benign_ratio}")
    x_resampled, y_resampled = apply_smote(x_train_norm, y_train, begnin_ratio=benign_ratio)

    print(f"Len of x_train_norm: {len(x_train_norm)} and len of x_resampled: {len(x_resampled)}")
    print("---------------------------------------------------------------------")

    # Initialize models
    rf_model = RandomForestClassifier(random_state=42)
    dt_model = DecisionTreeClassifier(random_state=42)
    lr_model = LogisticRegression(max_iter=1000)
    xgb_model = XGBClassifier(use_label_encoder=False, eval_metric='mlogloss')

    # Train and evaluate each model
    models = [rf_model, dt_model, lr_model, xgb_model]
    model_names = ['Random Forest', 'Decision Tree', 'Logistic Regression', 'XGBoost']

    for model, name in zip(models, model_names):
        # FIXME: Plot learning curves here for each model
        # plot_learning_curves(model, x_resampled, y_resampled, name, benign_ratio)

        model.fit(x_resampled, y_resampled)
        y_pred = model.predict(x_test_norm)

        # Calculate metrics
        accuracy = accuracy_score(y_test, y_pred)
        precision = precision_score(y_test, y_pred, average='macro')
        recall = recall_score(y_test, y_pred, average='macro')
        f1 = f1_score(y_test, y_pred, average='macro')

        print(f"{name} - Accuracy: {accuracy}, Precision: {precision}, Recall: {recall}, F1 Score: {f1}")

        # Compute confusion matrix
        cm = confusion_matrix(y_test, y_pred)

        # For multiclass, calculate metrics per class
        FP = cm.sum(axis=0) - np.diag(cm)  
        FN = cm.sum(axis=1) - np.diag(cm)
        TP = np.diag(cm)
        TN = cm.sum() - (FP + FN + TP)

        # Compute FPR and FNR for each class and store the average if needed
        FPR = FP / (FP + TN)
        FNR = FN / (TP + FN)

        # Calculate average FPR and FNR across all classes
        avg_FPR = np.mean(FPR)
        avg_FNR = np.mean(FNR)

        # Store results including average FPR and FNR
        results.append({
            "Benign Ratio": benign_ratio,
            "Model": name,
            "Average FPR": avg_FPR,
            "Average FNR": avg_FNR
        })


results_df = pd.DataFrame(results)

# Save results to a CSV file
results_df.to_csv("smote_results.csv", index=False)