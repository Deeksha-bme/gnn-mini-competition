import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import f1_score

# Load training data
train = pd.read_csv("data/processed/train.csv")
X = train.drop("next_role", axis=1)
y = train["next_role"]

# Split for local validation
X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2, random_state=42)

# Train baseline model
clf = RandomForestClassifier(n_estimators=100, random_state=42)
clf.fit(X_train, y_train)

# Evaluate on validation set
y_pred_val = clf.predict(X_val)
score = f1_score(y_val, y_pred_val, average='macro')
print(f"Validation Macro-F1 Score: {score:.4f}")

# Predict on test set
test = pd.read_csv("data/processed/test.csv")
test_preds = clf.predict(test)

# Save submission
submission = pd.DataFrame({
    "user_id": test["user_id"],
    "snapshot_id": test["snapshot_id"],
    "predicted_role": test_preds
})
submission.to_csv("submissions/deeksha.csv", index=False)
print("Submission saved to submissions/deeksha.csv")
