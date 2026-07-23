import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, roc_auc_score, confusion_matrix, precision_score, recall_score, f1_score
import matplotlib.pyplot as plt
import seaborn as sns

# Set random seed for reproducibility
np.random.seed(42)

# =============================================
# STEP 1: Create Synthetic Credit Dataset
# =============================================
# Features: income, debt_ratio, payment_history (months on time), credit_utilization, age, num_loans
n_samples = 1000

data = {
    'income': np.random.normal(50000, 15000, n_samples).clip(10000, 120000),
    'debt_ratio': np.random.beta(2, 5, n_samples) * 0.8,  # mostly low debt
    'payment_history': np.random.randint(0, 36, n_samples),  # months paid on time
    'credit_utilization': np.random.beta(3, 7, n_samples) * 0.9,
    'age': np.random.randint(21, 70, n_samples),
    'num_loans': np.random.poisson(2, n_samples).clip(0, 8),
}

# Create target: creditworthy (1 = good, 0 = bad)
# Simple rule-based target + noise
data['creditworthy'] = (
    (data['income'] > 40000) &
    (data['debt_ratio'] < 0.4) &
    (data['payment_history'] > 12) &
    (data['credit_utilization'] < 0.5)
).astype(int)

# Add some noise to target
noise = np.random.random(n_samples) < 0.15
data['creditworthy'] = np.where(noise, 1 - data['creditworthy'], data['creditworthy'])

df = pd.DataFrame(data)

print("Dataset Shape:", df.shape)
print("\nTarget Distribution:\n", df['creditworthy'].value_counts(normalize=True))

# =============================================
# STEP 2: Feature Engineering
# =============================================
df['income_to_debt'] = df['income'] / (df['debt_ratio'] + 0.01)  # avoid division by zero
df['payment_score'] = df['payment_history'] / (df['num_loans'] + 1)
df['age_group'] = pd.cut(df['age'], bins=[0, 30, 45, 60, 100], labels=[0, 1, 2, 3]).astype(int)

features = ['income', 'debt_ratio', 'payment_history', 'credit_utilization', 
            'age', 'num_loans', 'income_to_debt', 'payment_score', 'age_group']
X = df[features]
y = df['creditworthy']

# =============================================
# STEP 3: Train-Test Split + Scaling
# =============================================
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.25, random_state=42, stratify=y)

scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# =============================================
# STEP 4: Train Multiple Models
# =============================================
models = {
    "Logistic Regression": LogisticRegression(max_iter=1000, random_state=42),
    "Decision Tree": DecisionTreeClassifier(max_depth=6, random_state=42),
    "Random Forest": RandomForestClassifier(n_estimators=200, max_depth=8, random_state=42)
}

results = {}

for name, model in models.items():
    if name == "Logistic Regression":
        model.fit(X_train_scaled, y_train)
        y_pred = model.predict(X_test_scaled)
        y_pred_proba = model.predict_proba(X_test_scaled)[:, 1]
    else:
        model.fit(X_train, y_train)
        y_pred = model.predict(X_test)
        y_pred_proba = model.predict_proba(X_test)[:, 1]
    
    results[name] = {
        'precision': precision_score(y_test, y_pred),
        'recall': recall_score(y_test, y_pred),
        'f1': f1_score(y_test, y_pred),
        'roc_auc': roc_auc_score(y_test, y_pred_proba)
    }
    
    print(f"\n=== {name} ===")
    print(classification_report(y_test, y_pred))
    print(f"ROC-AUC: {results[name]['roc_auc']:.4f}")

# =============================================
# STEP 5: Compare Models
# =============================================
results_df = pd.DataFrame(results).T
print("\n=== Model Comparison ===")
print(results_df.round(4))

# =============================================
# STEP 6: Feature Importance (Random Forest)
# =============================================
rf = models["Random Forest"]
importances = pd.Series(rf.feature_importances_, index=features)
importances.sort_values(ascending=False).plot(kind='bar', figsize=(10, 6), title='Feature Importance')
plt.ylabel('Importance')
plt.tight_layout()
plt.show()

# Confusion Matrix for best model (Random Forest)
best_model = models["Random Forest"]
y_pred_best = best_model.predict(X_test)
cm = confusion_matrix(y_test, y_pred_best)

plt.figure(figsize=(6, 5))
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
            xticklabels=['Bad', 'Good'], yticklabels=['Bad', 'Good'])
plt.title('Confusion Matrix - Random Forest')
plt.ylabel('Actual')
plt.xlabel('Predicted')
plt.show()