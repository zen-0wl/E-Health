#!/usr/bin/env python
# coding: utf-8

# In[1]:


import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.ensemble import ExtraTreesClassifier
from sklearn.neighbors import KNeighborsClassifier
from xgboost import XGBClassifier
from imblearn.combine import SMOTEENN
from imblearn.over_sampling import SMOTE
import lightgbm as lgb
import matplotlib.pyplot as plt
import seaborn as sns


# <h5 style="color: SkyBlue;">Load Dataset</h5>

# In[2]:


df = pd.read_csv('user_data_for_disease_prediction - unclassified data set.csv')
print(df.head())


# In[3]:


# handle missing values 
df.fillna(0, inplace=True)


# <h5 style="color: SkyBlue;">Ranking of diseases in the dataset</h5>

# In[4]:


def classify_disease(row):
    gender = row["Gender (0-M;1-F)"]

    # Hypertension
    if (
        row["Blood Pressure (systolic)"] > 130 or row["Blood Pressure (diastolic)"] > 80
    ) and (row["Heart Rate (bpm)"] >= 60 and row["Heart Rate (bpm)"] <= 100):
        return "Hypertension"

    # Atherosclerosis (distinct criteria for this condition)
    elif row["Blood Pressure (systolic)"] > 140 or row["Hemoglobin A1c (%)"] > 7.0:
        return "Atherosclerosis"

    # Cardiovascular Disease (CVD)
    elif (row["Heart Rate (bpm)"] > 100 or row["Heart Rate (bpm)"] < 60) or (
        row["Blood Pressure (systolic)"] > 140 or row["Blood Pressure (diastolic)"] > 90
    ):
        return "Cardiovascular Disease (CVD)"

    # Respiratory Disease (COPD or Asthma)
    elif row["Breathing Rate (brpm)"] > 20 or row["Oxygen Saturation (%)"] < 90:
        return "Respiratory Disease (COPD or Asthma)"

    # Chronic Fatigue Syndrome (CFS)
    elif (
        row["HRV SDNN (ms)"] < 50
        or row["RMSSD (ms)"] < 30
        or row["Recovery Ability"] == 0
    ):
        return "Chronic Fatigue Syndrome (CFS)"

    # Diabetes
    elif row["Hemoglobin A1c (%)"] > 6.5:
        return "Diabetes"

    # Arrhythmias
    elif row["HRV SDNN (ms)"] > 100 or row["Mean RRI (ms)"] < 600:
        return "Arrhythmias"

    # Stress-related Disorders
    elif row["Stress Index"] > 70 or row["SNS Index"] > 1.0:
        return "Stress-related Disorders"

    # Autonomic Dysfunction
    elif row["PNS Index"] < -1.0 or row["SNS Index"] > 1.0:
        return "Autonomic Dysfunction"

    # Anaemia (adjusted for gender)
    elif (gender == 0 and row["Hemoglobin (g/dl)"] < 13.5) or (
        gender == 1 and row["Hemoglobin (g/dl)"] < 12.0
    ):
        return "Anaemia"

    # If none of the conditions are met, classify as Healthy
    else:
        return "Healthy"


# Apply the updated classify_disease function to each row and create a new column for the disease
df["Disease"] = df.apply(classify_disease, axis=1)

# Count occurrences of each disease
disease_counts = df["Disease"].value_counts()

# Sort counts in descending order and display the results
disease_counts_sorted = disease_counts.sort_values(ascending=False)
print(disease_counts_sorted)


# In[5]:


sampled_df_health = df.sample(n=200000, random_state=42)

# Count occurrences of each disease
sampled_disease_counts = sampled_df_health["Disease"].value_counts()

# Sort counts in descending order and display the results
sampled_disease_counts_sorted = sampled_disease_counts.sort_values(ascending=False)
print(sampled_disease_counts_sorted)


# <h5 style="color: SkyBlue;">Synthetic sampling</h5>

# In[6]:


def apply_smoteenn(sampled_df_health):
    """
    SMOTEENN sampling to balance the dataset.
    
    Parameters:
        sampled_df_health (DataFrame): Data containing features and the target column 'Disease'.
        
    Returns:
        X_balanced (DataFrame): Resampled features.
        y_balanced (Series): Resampled target labels.
        counts (dict): Summary of the balanced dataset.
    """
    # Feature (X) and target (y) separation
    X = sampled_df_health.drop("Disease", axis=1)
    y = sampled_df_health["Disease"]

    # Initialise SMOTE and SMOTEENN
    smote = SMOTE(k_neighbors=2, random_state=42)
    smote_enn = SMOTEENN(random_state=42, smote=smote)

    # Apply resampling
    X_balanced, y_balanced = smote_enn.fit_resample(X, y)

    # Summary of the balanced dataset
    counts = {
        "Balanced Feature Shape": X_balanced.shape,
        "Balanced Target Shape": y_balanced.shape,
        "Balanced Distribution": y_balanced.value_counts().to_dict(),
    }
    
    print(counts)  # Optional: Print the summary

    return X_balanced, y_balanced, counts


# <h5 style="color: SkyBlue;">Logistic Regression</h5>

# In[7]:

X = sampled_df_health.drop("Disease", axis=1)
y = sampled_df_health["Disease"]
smote = SMOTE(k_neighbors=2, random_state=42)
smote_enn = SMOTEENN(random_state=42, smote=smote)
X_balanced, y_balanced = smote_enn.fit_resample(X, y)
label_encoder = LabelEncoder()
y_encoded = label_encoder.fit_transform(y_balanced)  # Encode the balanced target

X_train, X_test, y_train, y_test = train_test_split(
    X_balanced, y_encoded, test_size=0.2, random_state=42
)

model = LogisticRegression(
    max_iter=1000, class_weight='balanced', multi_class='ovr', random_state=42
)
model.fit(X_train, y_train)

y_pred = model.predict(X_test)
y_test_decoded = label_encoder.inverse_transform(y_test)
y_pred_decoded = label_encoder.inverse_transform(y_pred)

print("Classification Report:\n", classification_report(y_test_decoded, y_pred_decoded))

accuracy = accuracy_score(y_test, y_pred)
print(f"Overall Accuracy: {accuracy:.2f}")

conf_matrix = confusion_matrix(y_test_decoded, y_pred_decoded, labels=label_encoder.classes_)

# confusion Matrix
plt.figure(figsize=(12, 8))
sns.heatmap(conf_matrix, annot=True, fmt='d', cmap='Blues',
            xticklabels=label_encoder.classes_, yticklabels=label_encoder.classes_)
plt.title('Confusion Matrix for Disease Classification', fontsize=16)
plt.xlabel('Predicted', fontsize=12)
plt.ylabel('Actual', fontsize=12)
plt.xticks(rotation=45, ha='right')
plt.yticks(rotation=0)
plt.tight_layout()
plt.show()


# <h5 style="color: SkyBlue;">Decision Tree</h5>

# In[8]:


dt = DecisionTreeClassifier(random_state=42)
dt.fit(X_train, y_train)

y_pred_dt = dt.predict(X_test)

print("Decision Tree - Classification Report")
print(classification_report(y_test, y_pred_dt))
print("Decision Tree - Accuracy:", accuracy_score(y_test, y_pred_dt))

conf_matrix_dt = confusion_matrix(y_test, y_pred_dt)
sns.heatmap(conf_matrix_dt, annot=True, cmap='Oranges', fmt='d',
            xticklabels=label_encoder.classes_, yticklabels=label_encoder.classes_)
plt.title('Decision Tree - Confusion Matrix')
plt.show()


# <h5 style="color: SkyBlue;">Random Forest</h5>

# In[9]:


rf = RandomForestClassifier(n_estimators=100, random_state=42)
rf.fit(X_train, y_train)

y_pred_rf = rf.predict(X_test)

print("Random Forest - Classification Report")
print(classification_report(y_test, y_pred_rf))
print("Random Forest - Accuracy:", accuracy_score(y_test, y_pred_rf))

conf_matrix_rf = confusion_matrix(y_test, y_pred_rf)
sns.heatmap(conf_matrix_rf, annot=True, cmap='YlGnBu', fmt='d',
            xticklabels=label_encoder.classes_, yticklabels=label_encoder.classes_)
plt.title('Random Forest - Confusion Matrix')
plt.show()


# <h5 style="color: SkyBlue;">GBM</h5>

# In[17]:


gbm = GradientBoostingClassifier(n_estimators=100, learning_rate=0.1, random_state=42)
gbm.fit(X_train, y_train)

y_pred_gbm = gbm.predict(X_test)

print("Gradient Boosting Model - Classification Report")
print(classification_report(y_test, y_pred_gbm))

print("Gradient Boosting Model - Accuracy:", accuracy_score(y_test, y_pred_gbm))

# confusion Matrix
conf_matrix_gbm = confusion_matrix(y_test, y_pred_gbm)
sns.heatmap(conf_matrix_gbm, annot=True, cmap='Blues', fmt='d', 
            xticklabels=label_encoder.classes_, yticklabels=label_encoder.classes_)
plt.title('Gradient Boosting - Confusion Matrix')
plt.show()


# <h5 style="color: SkyBlue;">KNN</h5>

# In[11]:


knn = KNeighborsClassifier(n_neighbors=5)
knn.fit(X_train, y_train)

y_pred_knn = knn.predict(X_test)

print("K-Nearest Neighbors Model - Classification Report")
print(classification_report(y_test, y_pred_knn))

print("K-Nearest Neighbors Model - Accuracy:", accuracy_score(y_test, y_pred_knn))

# confusion Matrix
conf_matrix_knn = confusion_matrix(y_test, y_pred_knn)
sns.heatmap(conf_matrix_knn, annot=True, cmap='Greens', fmt='d', 
            xticklabels=label_encoder.classes_, yticklabels=label_encoder.classes_)
plt.title('KNN - Confusion Matrix')
plt.show()


# <h5 style="color: SkyBlue;">LightGBM</h5>

# In[14]:


train_data = lgb.Dataset(X_train, label=y_train)

params = {
    'objective': 'binary',
    'metric': 'binary_error',
    'boosting_type': 'gbdt',
    'verbosity': -1
}
lgb_model = lgb.train(params, train_data, num_boost_round=100)

y_pred_lgb = (lgb_model.predict(X_test) >= 0.5).astype(int)

print("LightGBM - Classification Report")
print(classification_report(y_test, y_pred_lgb))
print("LightGBM - Accuracy:", accuracy_score(y_test, y_pred_lgb))

# confusion  matrix
conf_matrix_lgb = confusion_matrix(y_test, y_pred_lgb)

plt.figure(figsize=(8, 6))
sns.heatmap(conf_matrix_lgb, annot=True, cmap="Blues", fmt="d", cbar=False)
plt.title("LightGBM Confusion Matrix")
plt.xlabel("Predicted")
plt.ylabel("Actual")
plt.show()


# <h5 style="color: SkyBlue;">XGBoost</h5>

# In[15]:


xgb = XGBClassifier(use_label_encoder=False, eval_metric='mlogloss', random_state=42)
xgb.fit(X_train, y_train)

y_pred_xgb = xgb.predict(X_test)

print("XGBoost Model - Classification Report")
print(classification_report(y_test, y_pred_xgb))

print("XGBoost Model - Accuracy:", accuracy_score(y_test, y_pred_xgb))

# confusion Matrix
conf_matrix_xgb = confusion_matrix(y_test, y_pred_xgb)
sns.heatmap(conf_matrix_xgb, annot=True, cmap='Reds', fmt='d', 
            xticklabels=label_encoder.classes_, yticklabels=label_encoder.classes_)
plt.title('XGBoost - Confusion Matrix')
plt.show()


# <h5 style="color: SkyBlue;">ExtraTreesClassifier</h5>

# In[16]:


X = X.astype('float32')

et_model = ExtraTreesClassifier(n_estimators=100, random_state=42, n_jobs=1)
et_model.fit(X_train, y_train)

y_pred_et = et_model.predict(X_test)

print("ExtraTrees - Classification Report")
print(classification_report(y_test, y_pred_et))
print("ExtraTrees - Accuracy:", accuracy_score(y_test, y_pred_et))

conf_matrix_et = confusion_matrix(y_test, y_pred_et)

plt.figure(figsize=(8, 6))
sns.heatmap(conf_matrix_et, annot=True, cmap="Blues", fmt="d", cbar=False)
plt.title("ExtraTrees Confusion Matrix")
plt.xlabel("Predicted")
plt.ylabel("Actual")
plt.show()

