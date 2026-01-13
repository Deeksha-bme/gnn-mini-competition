import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import f1_score

# Load training and test data
train = pd.read_csv("data/processed/train.csv")
test = pd.read_csv("data/processed/test.csv")

# Set features & target
X = train.drop("next_role", axis=1)
y = train["next_role"]

# Train a simple baseline model
clf = RandomForestClassifier(n_estimators=100, random_state=42)
clf.fit(X, y)

# Predict on test
test_preds = clf.predict(test)

# Save submission
submission = pd.DataFrame({
    "user_id": test["user_id"],
    "snapshot_id": test["snapshot_id"],
    "predicted_role": test_preds
})
submission.to_csv("submissions/deeksha.csv", index=False)
print("Saved submission to submissions/deeksha.csv")
