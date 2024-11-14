import pickle
import pandas as pd
from classify import classify_disease
from imblearn.over_sampling import SMOTE

df = pd.read_csv('user_data_for_disease_prediction - unclassified data set.csv')

df["Disease"] = df.apply(classify_disease, axis=1)

disease_counts = df["Disease"].value_counts()

disease_counts_sorted = disease_counts.sort_values(ascending=False)
print(disease_counts_sorted)

# Feature (X) and target (y) separation
X = df.drop("Disease", axis=1)
y = df["Disease"]

smote = SMOTE(random_state=42)
X_balanced, y_balanced = smote.fit_resample(X, y)

with open("balanced_data.pkl", "wb") as f:
    pickle.dump((X_balanced, y_balanced), f)

balanced_data_counts = {
    "Balanced Feature Shape ": X_balanced.shape,
    "Balanced Target Shape ": y_balanced.shape,
    "Balanced Distribution after SMOTEENN ": y_balanced.value_counts()
}

balanced_data_counts