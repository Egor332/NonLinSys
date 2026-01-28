from HIVPredictor import HIVPredictor
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import (
    confusion_matrix,
    classification_report,
    roc_auc_score,
    roc_curve,
    precision_recall_curve,
    average_precision_score,
    matthews_corrcoef,
    accuracy_score,
    f1_score
)
from tqdm import tqdm

testset_path = "dataset/HIV_test.csv"
model_path = "models/GeLU/c1_h4_l3_wS_oA_lB_e15_emb256.pth"
output_plot_path = "evaluation_results.png"

ACTIVITI_THRESHOLD = 0.5

print("Loading model and dataset...")
predictor = HIVPredictor(model_path)
df = pd.read_csv(testset_path)


y_true = []
y_pred_bin = []
y_scores = []
invalid_count = 0

print("Running predictions...")
for index, row in tqdm(df.iterrows(), total=df.shape[0]):
    smiles = row['smiles']

    valid_class = int(row['HIV_active'])

    model_output = predictor.predict(smiles)

    if 'error' in model_output:
        invalid_count += 1
        continue


    prob_active = float(model_output['confidence_active'])
    is_active_pred = 0
    if prob_active > ACTIVITI_THRESHOLD:
        is_active_pred = 1

    y_true.append(valid_class)
    y_pred_bin.append(is_active_pred)
    y_scores.append(prob_active)


y_true = np.array(y_true)
y_pred_bin = np.array(y_pred_bin)
y_scores = np.array(y_scores)


cm = confusion_matrix(y_true, y_pred_bin)
tn, fp, fn, tp = cm.ravel()


accuracy = accuracy_score(y_true, y_pred_bin)
mcc = matthews_corrcoef(y_true, y_pred_bin)
roc_auc = roc_auc_score(y_true, y_scores)
pr_auc = average_precision_score(y_true, y_scores)
f1 = f1_score(y_true, y_pred_bin)


sensitivity = tp / (tp + fn) if (tp + fn) > 0 else 0
specificity = tn / (tn + fp) if (tn + fp) > 0 else 0


print("\n" + "=" * 40)
print("       EVALUATION RESULTS")
print("=" * 40)
print(f"Total Samples Processed: {len(y_true)}")
print(f"Invalid SMILES Skipped:  {invalid_count}")
print("-" * 40)
print(f"Confusion Matrix:")
print(f" [ TP: {tp:<5} | FP: {fp:<5} ] (Predicted Active)")
print(f" [ FN: {fn:<5} | TN: {tn:<5} ] (Predicted Inactive)")
print("-" * 40)
print(f"Accuracy:        {accuracy:.4f}")
print(f"ROC-AUC:         {roc_auc:.4f} ")
print(f"PR-AUC:          {pr_auc:.4f}")
print(f"MCC:             {mcc:.4f}")
print(f"F1-Score:        {f1:.4f}")
print(f"Sensitivity (Recall): {sensitivity:.4f}")
print(f"Specificity:          {specificity:.4f}")
print("=" * 40)
print("\nFull Classification Report:")
print(classification_report(y_true, y_pred_bin, target_names=['Inactive', 'Active']))

plt.figure(figsize=(18, 5))

plt.subplot(1, 3, 1)
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', cbar=False,
            xticklabels=['Pred Inactive', 'Pred Active'],
            yticklabels=['True Inactive', 'True Active'])
plt.title(f'Confusion Matrix\nMCC: {mcc:.3f}')
plt.ylabel('Actual')
plt.xlabel('Predicted')

# Plot 2: ROC Curve
fpr, tpr, _ = roc_curve(y_true, y_scores)
plt.subplot(1, 3, 2)
plt.plot(fpr, tpr, color='darkorange', lw=2, label=f'ROC curve (AUC = {roc_auc:.2f})')
plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
plt.xlim([0.0, 1.0])
plt.ylim([0.0, 1.05])
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('Receiver Operating Characteristic (ROC)')
plt.legend(loc="lower right")
plt.grid(alpha=0.3)

# Plot 3: Precision-Recall Curve
precision, recall, _ = precision_recall_curve(y_true, y_scores)
plt.subplot(1, 3, 3)
plt.plot(recall, precision, color='green', lw=2, label=f'PR curve (AUC = {pr_auc:.2f})')
plt.xlabel('Recall (Sensitivity)')
plt.ylabel('Precision')
plt.title('Precision-Recall Curve')
plt.legend(loc="best")
plt.grid(alpha=0.3)

plt.tight_layout()
plt.savefig(output_plot_path)
print(f"\nPlots saved to {output_plot_path}")
plt.show()