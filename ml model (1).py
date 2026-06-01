import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import warnings
warnings.filterwarnings('ignore')

from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    accuracy_score, classification_report,
    confusion_matrix, roc_curve, auc
)

# ── load data ────────────────────────────────────────────────
url = "https://raw.githubusercontent.com/datasciencedojo/datasets/master/titanic.csv"
df = pd.read_csv(url)

# ── quick cleaning (same logic as before) ───────────────────
df['Age'] = df.groupby(['Pclass', 'Sex'])['Age'].transform(
    lambda x: x.fillna(x.median())
)
df['Embarked'].fillna(df['Embarked'].mode()[0], inplace=True)
df.drop(columns=['Cabin', 'Ticket', 'Name', 'PassengerId'], inplace=True)
df.drop_duplicates(inplace=True)

for col in ['Age', 'Fare']:
    q1, q3 = df[col].quantile(0.25), df[col].quantile(0.75)
    iqr = q3 - q1
    df[col] = df[col].clip(q1 - 1.5*iqr, q3 + 1.5*iqr)

# feature engineering
df['family_size'] = df['SibSp'] + df['Parch'] + 1
df['alone'] = (df['family_size'] == 1).astype(int)

# encode categorical columns
le = LabelEncoder()
df['Sex'] = le.fit_transform(df['Sex'])           # male=1, female=0
df['Embarked'] = le.fit_transform(df['Embarked']) # C=0, Q=1, S=2

# ── features & target ────────────────────────────────────────
features = ['Pclass', 'Sex', 'Age', 'Fare', 'SibSp', 'Parch',
            'Embarked', 'family_size', 'alone']
X = df[features]
y = df['Survived']

# train/test split — 80/20
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

# scale features (needed for logistic regression)
scaler = StandardScaler()
X_train_sc = scaler.fit_transform(X_train)
X_test_sc  = scaler.transform(X_test)

print(f"Train size: {X_train.shape[0]}  |  Test size: {X_test.shape[0]}")

# ── train models ─────────────────────────────────────────────
models = {
    'Logistic Regression': LogisticRegression(max_iter=500),
    'Decision Tree':       DecisionTreeClassifier(max_depth=5, random_state=42),
    'Random Forest':       RandomForestClassifier(n_estimators=100, random_state=42)
}

results = {}

print("\n--- model results ---")
for name, model in models.items():
    # logistic regression uses scaled data
    if name == 'Logistic Regression':
        model.fit(X_train_sc, y_train)
        y_pred = model.predict(X_test_sc)
        y_prob = model.predict_proba(X_test_sc)[:, 1]
        cv_scores = cross_val_score(model, X_train_sc, y_train, cv=5)
    else:
        model.fit(X_train, y_train)
        y_pred = model.predict(X_test)
        y_prob = model.predict_proba(X_test)[:, 1]
        cv_scores = cross_val_score(model, X_train, y_train, cv=5)

    acc = accuracy_score(y_test, y_pred)
    results[name] = {
        'model': model,
        'y_pred': y_pred,
        'y_prob': y_prob,
        'accuracy': acc,
        'cv_mean': cv_scores.mean(),
        'cv_std': cv_scores.std(),
        'cm': confusion_matrix(y_test, y_pred)
    }

    print(f"\n{name}")
    print(f"  test accuracy : {acc*100:.2f}%")
    print(f"  5-fold CV     : {cv_scores.mean()*100:.2f}% (+/- {cv_scores.std()*100:.2f}%)")
    print(classification_report(y_test, y_pred, target_names=['Died', 'Survived']))

# ── feature importance (random forest) ───────────────────────
rf = results['Random Forest']['model']
importances = pd.Series(rf.feature_importances_, index=features).sort_values(ascending=False)

# ── plots ────────────────────────────────────────────────────
fig, axes = plt.subplots(3, 3, figsize=(15, 13))
fig.suptitle('ML Model Evaluation — Titanic Survival Prediction',
             fontsize=15, fontweight='bold', y=1.01)

colors = {'Logistic Regression': '#3498db',
          'Decision Tree':       '#e67e22',
          'Random Forest':       '#2ecc71'}

# row 0: confusion matrices
for i, (name, res) in enumerate(results.items()):
    ax = axes[0, i]
    sns.heatmap(res['cm'], annot=True, fmt='d', cmap='Blues', ax=ax,
                xticklabels=['Died', 'Survived'],
                yticklabels=['Died', 'Survived'],
                linewidths=0.5, cbar=False)
    ax.set_title(f'Confusion Matrix\n{name}', fontsize=10, fontweight='bold')
    ax.set_ylabel('Actual')
    ax.set_xlabel('Predicted')

# row 1 col 0-1: ROC curves (all 3 on one plot)
ax = axes[1, 0]
for name, res in results.items():
    fpr, tpr, _ = roc_curve(y_test, res['y_prob'])
    roc_auc = auc(fpr, tpr)
    ax.plot(fpr, tpr, label=f"{name} (AUC={roc_auc:.2f})",
            color=colors[name], linewidth=2)
ax.plot([0,1], [0,1], 'k--', linewidth=1)
ax.set_title('ROC Curves', fontweight='bold')
ax.set_xlabel('False Positive Rate')
ax.set_ylabel('True Positive Rate')
ax.legend(fontsize=8)
ax.set_xlim([0, 1])
ax.set_ylim([0, 1.02])

# row 1 col 1: accuracy comparison bar chart
ax = axes[1, 1]
names = list(results.keys())
accs  = [results[n]['accuracy'] * 100 for n in names]
short = ['LR', 'DT', 'RF']
bars  = ax.bar(short, accs, color=[colors[n] for n in names], width=0.4)
ax.set_title('Test Accuracy Comparison (%)', fontweight='bold')
ax.set_ylim(60, 100)
ax.set_ylabel('Accuracy (%)')
for bar, val in zip(bars, accs):
    ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.3,
            f'{val:.1f}%', ha='center', va='bottom', fontsize=9)

# row 1 col 2: cross-val scores
ax = axes[1, 2]
cv_means = [results[n]['cv_mean'] * 100 for n in names]
cv_stds  = [results[n]['cv_std'] * 100 for n in names]
ax.bar(short, cv_means, yerr=cv_stds, color=[colors[n] for n in names],
       width=0.4, capsize=5, error_kw={'linewidth': 1.5})
ax.set_title('5-Fold Cross-Validation Accuracy', fontweight='bold')
ax.set_ylim(60, 100)
ax.set_ylabel('CV Accuracy (%)')

# row 2 col 0: feature importances
ax = axes[2, 0]
importances.plot(kind='bar', ax=ax, color='#2ecc71', edgecolor='none')
ax.set_title('Feature Importances (Random Forest)', fontweight='bold')
ax.set_ylabel('Importance')
ax.tick_params(axis='x', rotation=30)

# row 2 col 1: precision / recall per model
ax = axes[2, 1]
from sklearn.metrics import precision_score, recall_score, f1_score
metrics_data = {
    'Precision': [precision_score(y_test, results[n]['y_pred']) for n in names],
    'Recall':    [recall_score(y_test, results[n]['y_pred'])    for n in names],
    'F1 Score':  [f1_score(y_test, results[n]['y_pred'])        for n in names],
}
x = np.arange(len(short))
width = 0.25
for j, (metric, vals) in enumerate(metrics_data.items()):
    ax.bar(x + j*width, vals, width, label=metric)
ax.set_xticks(x + width)
ax.set_xticklabels(short)
ax.set_title('Precision / Recall / F1', fontweight='bold')
ax.set_ylim(0, 1)
ax.legend(fontsize=8)

# row 2 col 2: prediction distribution (RF)
ax = axes[2, 2]
rf_probs = results['Random Forest']['y_prob']
ax.hist(rf_probs[y_test == 0], bins=20, alpha=0.6, color='#e74c3c', label='Actual: Died')
ax.hist(rf_probs[y_test == 1], bins=20, alpha=0.6, color='#2ecc71', label='Actual: Survived')
ax.axvline(0.5, color='black', linestyle='--', linewidth=1)
ax.set_title('RF Predicted Probability Distribution', fontweight='bold')
ax.set_xlabel('Predicted Probability of Survival')
ax.set_ylabel('Count')
ax.legend(fontsize=8)

plt.tight_layout()
plt.savefig('ml_dashboard.png', dpi=150, bbox_inches='tight')
print("\nsaved → ml_dashboard.png")
plt.show()

# ── best model summary ───────────────────────────────────────
best = max(results, key=lambda n: results[n]['accuracy'])
print(f"\nbest model: {best} ({results[best]['accuracy']*100:.2f}% accuracy)")
print(f"top 3 features: {', '.join(importances.head(3).index.tolist())}")
