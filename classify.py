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