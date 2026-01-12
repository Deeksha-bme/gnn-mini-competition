import os
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import f1_score

# -----------------------
# 1️⃣ Paths to files
# -----------------------
repo_root = os.path.dirname(os.path.abspath(__file__))

train_file = os.path.join(repo_root, "data", "processed", "train.csv")
test_file = os.path.join(repo_root, "data", "processed", "test.csv")
submission_file = os.path.join(repo_root, "submissions", "deeksha.csv")

# -----------------------
# 2️⃣ Load training and test data
# -----------------------
train = pd.read_csv(train_file)
X = train.drop("next_role", axis=1)
y = train["next_role"]

test = pd.read_csv(test_file)
X_test = test.copy()

# -----------------------
# 3️⃣ Split training data for local validation
# -----------------------
X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2, random_state=42)

# -----------------------
# 4️⃣ Train RandomForest baseline
# -----------------------
clf = RandomForestClassifier(n_estimators=100, random_state=42)
clf.fit(X_train, y_train)

# -----------------------
# 5️⃣ Evaluate locally (optional)
# -----------------------
y_val_pred = clf.predict(X_val)
val_score = f1_score(y_val, y_val_pred, average="macro")
print(f"Validation Macro-F1 Score: {val_score:.4f}")

# -----------------------
# 6️⃣ Predict on test set
# -----------------------
test_preds = clf.predict(X_test)

submission = pd.DataFrame({
    "user_id": X_test["user_id"],
    "snapshot_id": X_test["snapshot_id"],
    "predicted_role": test_preds
})

submission.to_csv(submission_file, index=False)
print(f"✅ Submission saved to {submission_file}")
