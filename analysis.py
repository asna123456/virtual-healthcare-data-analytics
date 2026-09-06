"""
Virtual Healthcare Data Analytics
YuvaIntern Internship Program

Project:
Predictive Analytics and Risk-Pattern Detection for Early Identification
of Health Deterioration in Remote Heart Failure Monitoring

Dataset:
PerHeart Pilot Dataset - Physiological Data from Older Adults with
Heart Failure

Purpose:
- Load and inspect healthcare datasets
- Perform basic data-quality checks
- Clean duplicate records and timestamps
- Aggregate physiological measurements at patient-day level
- Create day-to-day trend features
- Create patient-specific baseline and deviation features
- Generate an exploratory, non-clinical risk indicator

IMPORTANT:
The dataset contains a small cohort and does not provide a ground-truth
clinical deterioration label. Therefore, the deviation-based indicator
created in this script is exploratory only. It is NOT a validated clinical
risk score and must NOT be interpreted as a diagnosis or confirmed
health deterioration.
"""

import pandas as pd


# ============================================================
# 1. LOAD THE PERHEART MEDICAL DATASETS
# ============================================================

print("\n" + "=" * 60)
print("LOADING PERHEART MEDICAL DATASETS")
print("=" * 60)

blood_pressure = pd.read_csv("medical/blood_pressure.csv")
body_mass = pd.read_csv("medical/body_mass.csv")
glucose = pd.read_csv("medical/glucose.csv")
oxidation = pd.read_csv("medical/oxidation.csv")
temperature = pd.read_csv("medical/temperature.csv")

personal_questionnaires = pd.read_csv(
    "medical/personal_questionnaires.csv"
)


# ============================================================
# 2. STORE MEDICAL DATASETS IN A DICTIONARY
# ============================================================

medical_datasets = {
    "Blood Pressure": blood_pressure,
    "Body Mass": body_mass,
    "Glucose": glucose,
    "Oxidation / SpO2": oxidation,
    "Temperature": temperature
}


# ============================================================
# 3. DISPLAY BASIC DATASET INFORMATION
# ============================================================

print("\n" + "=" * 60)
print("BASIC DATASET INFORMATION")
print("=" * 60)

for name, df in medical_datasets.items():

    print("\n" + "-" * 50)
    print(name)
    print("-" * 50)

    print("Shape:", df.shape)
    print("Number of patients:", df["user_id"].nunique())
    print("Columns:", list(df.columns))

    print("\nFirst 5 rows:")
    print(df.head())

    print("\nMissing values:")
    print(df.isnull().sum())

    print("\nDuplicate rows:", df.duplicated().sum())


# ============================================================
# 4. PERSONAL QUESTIONNAIRE INFORMATION
# ============================================================

print("\n" + "=" * 60)
print("PERSONAL QUESTIONNAIRE")
print("=" * 60)

print("Shape:", personal_questionnaires.shape)

print("\nColumns:")
print(list(personal_questionnaires.columns))

print("\nFirst 5 rows:")
print(personal_questionnaires.head())

print("\nMissing values:")
print(personal_questionnaires.isnull().sum())


# ============================================================
# 5. CONVERT UNIX TIMESTAMPS TO DATETIME
# ============================================================

print("\n" + "=" * 60)
print("TIMESTAMP CONVERSION")
print("=" * 60)

for name, df in medical_datasets.items():

    df["datetime"] = pd.to_datetime(
        df["ts"],
        unit="s",
        errors="coerce"
    )

    print(
        f"{name}: "
        f"{df['datetime'].notna().sum()} valid timestamps, "
        f"{df['datetime'].isna().sum()} invalid timestamps"
    )


# ============================================================
# 6. CHECK DATE RANGES
# ============================================================

print("\n" + "=" * 60)
print("DATE RANGES")
print("=" * 60)

for name, df in medical_datasets.items():

    print(
        name,
        "->",
        df["datetime"].min(),
        "to",
        df["datetime"].max()
    )


# ============================================================
# 7. PATIENT INFORMATION
# ============================================================

print("\n" + "=" * 60)
print("PATIENT INFORMATION")
print("=" * 60)

for name, df in medical_datasets.items():

    patients = sorted(
        df["user_id"].dropna().unique()
    )

    print(f"\n{name}:")
    print("Number of unique patients:", len(patients))
    print("Patient IDs:", patients)


# ============================================================
# 8. DATA CLEANING
# ============================================================

print("\n" + "=" * 60)
print("DATA CLEANING")
print("=" * 60)


# ------------------------------------------------------------
# Remove duplicate temperature records
# ------------------------------------------------------------

temperature_before = len(temperature)

temperature = temperature.drop_duplicates()

temperature_after = len(temperature)

temperature_duplicates_removed = (
    temperature_before - temperature_after
)

print(
    f"\nTemperature records before duplicate removal: "
    f"{temperature_before}"
)

print(
    f"Temperature records after duplicate removal: "
    f"{temperature_after}"
)

print(
    f"Duplicate temperature records removed: "
    f"{temperature_duplicates_removed}"
)


# ------------------------------------------------------------
# Re-create medical dataset dictionary after cleaning
# ------------------------------------------------------------

medical_datasets = {
    "Blood Pressure": blood_pressure,
    "Body Mass": body_mass,
    "Glucose": glucose,
    "Oxidation / SpO2": oxidation,
    "Temperature": temperature
}


# ============================================================
# 9. RE-CHECK TIMESTAMPS AFTER CLEANING
# ============================================================

print("\n" + "=" * 60)
print("INVALID TIMESTAMP CHECK")
print("=" * 60)

for name, df in medical_datasets.items():

    df["datetime"] = pd.to_datetime(
        df["ts"],
        unit="s",
        errors="coerce"
    )

    invalid_timestamps = df["datetime"].isna().sum()

    print(
        f"{name}: {invalid_timestamps} invalid timestamps"
    )


# ============================================================
# 10. SORT DATA BY PATIENT AND TIME
# ============================================================

for name, df in medical_datasets.items():

    df.sort_values(
        by=["user_id", "datetime"],
        inplace=True
    )


print("\nData cleaning completed successfully.")


# ============================================================
# 11. DAILY FEATURE ENGINEERING
# ============================================================

print("\n" + "=" * 60)
print("DAILY HEALTH FEATURE ENGINEERING")
print("=" * 60)


# ------------------------------------------------------------
# Blood Pressure
# ------------------------------------------------------------

bp_daily = (
    blood_pressure
    .groupby(
        [
            "user_id",
            blood_pressure["datetime"].dt.date
        ]
    )
    .agg(
        avg_systolic=("bp_sys", "mean"),
        avg_diastolic=("bp_dia", "mean"),
        avg_heart_rate=("hr", "mean"),
        bp_measurements=("bp_sys", "count")
    )
    .reset_index()
)

bp_daily.rename(
    columns={"datetime": "date"},
    inplace=True
)


# ------------------------------------------------------------
# Body Mass
# ------------------------------------------------------------

weight_daily = (
    body_mass
    .groupby(
        [
            "user_id",
            body_mass["datetime"].dt.date
        ]
    )
    .agg(
        avg_weight=("value", "mean"),
        weight_measurements=("value", "count")
    )
    .reset_index()
)

weight_daily.rename(
    columns={"datetime": "date"},
    inplace=True
)


# ------------------------------------------------------------
# SpO2
# ------------------------------------------------------------

spo2_daily = (
    oxidation
    .groupby(
        [
            "user_id",
            oxidation["datetime"].dt.date
        ]
    )
    .agg(
        avg_spo2=("sat", "mean"),
        min_spo2=("sat", "min"),
        avg_spo2_heart_rate=("hr", "mean"),
        spo2_measurements=("sat", "count")
    )
    .reset_index()
)

spo2_daily.rename(
    columns={"datetime": "date"},
    inplace=True
)


# ------------------------------------------------------------
# Temperature
# ------------------------------------------------------------

temperature_daily = (
    temperature
    .groupby(
        [
            "user_id",
            temperature["datetime"].dt.date
        ]
    )
    .agg(
        avg_temperature=("value", "mean"),
        min_temperature=("value", "min"),
        max_temperature=("value", "max"),
        temperature_measurements=("value", "count")
    )
    .reset_index()
)

temperature_daily.rename(
    columns={"datetime": "date"},
    inplace=True
)


# ------------------------------------------------------------
# Glucose
# ------------------------------------------------------------

glucose_daily = (
    glucose
    .groupby(
        [
            "user_id",
            glucose["datetime"].dt.date
        ]
    )
    .agg(
        avg_glucose=("value", "mean"),
        glucose_measurements=("value", "count")
    )
    .reset_index()
)

glucose_daily.rename(
    columns={"datetime": "date"},
    inplace=True
)


# ============================================================
# 12. DISPLAY DAILY FEATURES
# ============================================================

print("\n===== BLOOD PRESSURE DAILY =====")
print(bp_daily.head())

print("\n===== BODY MASS DAILY =====")
print(weight_daily.head())

print("\n===== SpO2 DAILY =====")
print(spo2_daily.head())

print("\n===== TEMPERATURE DAILY =====")
print(temperature_daily.head())

print("\n===== GLUCOSE DAILY =====")
print(glucose_daily.head())


# ============================================================
# 13. CREATE DAY-TO-DAY TREND FEATURES
# ============================================================

print("\n" + "=" * 60)
print("TREND FEATURE ENGINEERING")
print("=" * 60)


# Sort daily datasets

bp_daily = bp_daily.sort_values(
    ["user_id", "date"]
)

weight_daily = weight_daily.sort_values(
    ["user_id", "date"]
)

spo2_daily = spo2_daily.sort_values(
    ["user_id", "date"]
)

temperature_daily = temperature_daily.sort_values(
    ["user_id", "date"]
)

glucose_daily = glucose_daily.sort_values(
    ["user_id", "date"]
)


# ------------------------------------------------------------
# Blood pressure trends
# ------------------------------------------------------------

bp_daily["systolic_change"] = (
    bp_daily
    .groupby("user_id")["avg_systolic"]
    .diff()
)

bp_daily["diastolic_change"] = (
    bp_daily
    .groupby("user_id")["avg_diastolic"]
    .diff()
)

bp_daily["heart_rate_change"] = (
    bp_daily
    .groupby("user_id")["avg_heart_rate"]
    .diff()
)


# ------------------------------------------------------------
# Weight trend
# ------------------------------------------------------------

weight_daily["weight_change"] = (
    weight_daily
    .groupby("user_id")["avg_weight"]
    .diff()
)


# ------------------------------------------------------------
# SpO2 trend
# ------------------------------------------------------------

spo2_daily["spo2_change"] = (
    spo2_daily
    .groupby("user_id")["avg_spo2"]
    .diff()
)


# ------------------------------------------------------------
# Temperature trend
# ------------------------------------------------------------

temperature_daily["temperature_change"] = (
    temperature_daily
    .groupby("user_id")["avg_temperature"]
    .diff()
)


# ------------------------------------------------------------
# Glucose trend
# ------------------------------------------------------------

glucose_daily["glucose_change"] = (
    glucose_daily
    .groupby("user_id")["avg_glucose"]
    .diff()
)


print("Trend features created successfully.")


# ============================================================
# 14. CREATE FINAL PATIENT-DAY DATASET
# ============================================================

print("\n" + "=" * 60)
print("CREATING FINAL PATIENT-DAY DATASET")
print("=" * 60)


# Start with blood pressure data

final_daily = bp_daily.copy()


# Merge body weight

final_daily = final_daily.merge(
    weight_daily,
    on=["user_id", "date"],
    how="outer"
)


# Merge SpO2

final_daily = final_daily.merge(
    spo2_daily,
    on=["user_id", "date"],
    how="outer"
)


# Merge temperature

final_daily = final_daily.merge(
    temperature_daily,
    on=["user_id", "date"],
    how="outer"
)


# Merge glucose

final_daily = final_daily.merge(
    glucose_daily,
    on=["user_id", "date"],
    how="outer"
)


# Sort final dataset

final_daily = (
    final_daily
    .sort_values(
        ["user_id", "date"]
    )
    .reset_index(drop=True)
)


# ============================================================
# 15. FINAL DATASET INFORMATION
# ============================================================

print("\n" + "=" * 60)
print("FINAL PATIENT-DAY DATASET")
print("=" * 60)

print("\nDataset shape:")
print(final_daily.shape)

print("\nNumber of patients:")
print(final_daily["user_id"].nunique())

print("\nDate range:")
print(
    final_daily["date"].min(),
    "to",
    final_daily["date"].max()
)

print("\nColumns:")
print(list(final_daily.columns))


# ============================================================
# 16. MISSING VALUES IN FINAL DATASET
# ============================================================

print("\nMissing values:")
print(final_daily.isnull().sum())


# ============================================================
# 17. DISPLAY FIRST 10 ROWS
# ============================================================

print("\n===== FIRST 10 PATIENT-DAY RECORDS =====")

print(
    final_daily.head(10)
)


# ============================================================
# 18. SAVE FINAL PATIENT-DAY DATASET
# ============================================================

final_daily.to_csv(
    "final_patient_day_dataset.csv",
    index=False
)

print(
    "\nFinal patient-day dataset saved as:"
    " final_patient_day_dataset.csv"
)


# ============================================================
# 19. PATIENT-SPECIFIC BASELINES
# ============================================================

print("\n" + "=" * 60)
print("PATIENT-SPECIFIC BASELINES")
print("=" * 60)


# Calculate reference values for each patient

baseline = (
    final_daily
    .groupby("user_id")
    .agg(
        avg_systolic=("avg_systolic", "mean"),
        avg_diastolic=("avg_diastolic", "mean"),
        avg_heart_rate=("avg_heart_rate", "mean"),
        avg_weight=("avg_weight", "mean"),
        avg_spo2=("avg_spo2", "mean"),
        avg_temperature=("avg_temperature", "mean"),
        avg_glucose=("avg_glucose", "mean")
    )
    .reset_index()
)


print("\n===== PATIENT BASELINES =====")
print(baseline.head(10))


# ============================================================
# 20. MERGE BASELINES WITH DAILY DATA
# ============================================================

risk_data = final_daily.merge(
    baseline,
    on="user_id",
    suffixes=("", "_baseline")
)


# ============================================================
# 21. CALCULATE DEVIATIONS FROM PATIENT BASELINE
# ============================================================

print("\n" + "=" * 60)
print("PATIENT-SPECIFIC DEVIATION FEATURES")
print("=" * 60)


# Blood pressure deviation

risk_data["bp_deviation"] = (
    risk_data["avg_systolic"]
    - risk_data["avg_systolic_baseline"]
).abs()


# Heart-rate deviation

risk_data["heart_rate_deviation"] = (
    risk_data["avg_heart_rate"]
    - risk_data["avg_heart_rate_baseline"]
).abs()


# Weight deviation

risk_data["weight_deviation"] = (
    risk_data["avg_weight"]
    - risk_data["avg_weight_baseline"]
).abs()


# SpO2 directional deviation
# Positive values indicate that current SpO2 is below
# the patient's reference baseline.

risk_data["spo2_deviation"] = (
    risk_data["avg_spo2_baseline"]
    - risk_data["avg_spo2"]
)


# Temperature deviation

risk_data["temperature_deviation"] = (
    risk_data["avg_temperature"]
    - risk_data["avg_temperature_baseline"]
).abs()


# Glucose deviation

risk_data["glucose_deviation"] = (
    risk_data["avg_glucose"]
    - risk_data["avg_glucose_baseline"]
).abs()


print("Deviation features created successfully.")


# ============================================================
# 22. DISPLAY DEVIATION FEATURES
# ============================================================

print("\n===== SAMPLE DEVIATION FEATURES =====")

print(
    risk_data[
        [
            "user_id",
            "date",
            "bp_deviation",
            "heart_rate_deviation",
            "weight_deviation",
            "spo2_deviation",
            "temperature_deviation",
            "glucose_deviation"
        ]
    ].head(10)
)


# ============================================================
# 23. DEVIATION FEATURE SUMMARY
# ============================================================

print("\n" + "=" * 60)
print("DEVIATION FEATURE SUMMARY")
print("=" * 60)

deviation_features = [
    "bp_deviation",
    "heart_rate_deviation",
    "weight_deviation",
    "spo2_deviation",
    "temperature_deviation",
    "glucose_deviation"
]

print(
    risk_data[deviation_features].describe()
)


# ============================================================
# 24. SAVE PATIENT RISK FEATURES
# ============================================================

risk_data.to_csv(
    "patient_risk_features.csv",
    index=False
)

print(
    "\nPatient deviation features saved as:"
    " patient_risk_features.csv"
)


# ============================================================
# 25. EXPLORATORY DEVIATION-BASED RISK INDICATOR
# ============================================================

print("\n" + "=" * 60)
print("EXPLORATORY DEVIATION-BASED RISK INDICATOR")
print("=" * 60)


"""
IMPORTANT METHODOLOGICAL NOTE:

The following indicator is exploratory only.

It compares patient-day deviations against the median deviation
observed in the dataset. It is NOT a clinically validated risk score.

It should only be used to rank unusual patient-day patterns for
further exploratory analysis.

It does not represent:
- A medical diagnosis
- Confirmed health deterioration
- A validated clinical prediction
- A substitute for clinical assessment
"""


# ------------------------------------------------------------
# Calculate relative deviation components
# ------------------------------------------------------------

bp_median = risk_data["bp_deviation"].median()

heart_rate_median = (
    risk_data["heart_rate_deviation"].median()
)

weight_median = (
    risk_data["weight_deviation"].median()
)

positive_spo2_deviation = (
    risk_data["spo2_deviation"].clip(lower=0)
)

spo2_median = positive_spo2_deviation.median()

temperature_median = (
    risk_data["temperature_deviation"].median()
)


# ------------------------------------------------------------
# Avoid division by zero
# ------------------------------------------------------------

if bp_median > 0:

    risk_data["bp_risk"] = (
        risk_data["bp_deviation"] / bp_median
    )

else:

    risk_data["bp_risk"] = pd.NA


if heart_rate_median > 0:

    risk_data["heart_rate_risk"] = (
        risk_data["heart_rate_deviation"]
        / heart_rate_median
    )

else:

    risk_data["heart_rate_risk"] = pd.NA


if weight_median > 0:

    risk_data["weight_risk"] = (
        risk_data["weight_deviation"]
        / weight_median
    )

else:

    risk_data["weight_risk"] = pd.NA


if spo2_median > 0:

    risk_data["spo2_risk"] = (
        positive_spo2_deviation / spo2_median
    )

else:

    risk_data["spo2_risk"] = pd.NA


if temperature_median > 0:

    risk_data["temperature_risk"] = (
        risk_data["temperature_deviation"]
        / temperature_median
    )

else:

    risk_data["temperature_risk"] = pd.NA


# ------------------------------------------------------------
# Replace infinite values
# ------------------------------------------------------------

risk_data = risk_data.replace(
    [float("inf"), -float("inf")],
    pd.NA
)


# ------------------------------------------------------------
# Calculate exploratory overall indicator
# ------------------------------------------------------------

risk_columns = [
    "bp_risk",
    "heart_rate_risk",
    "weight_risk",
    "spo2_risk",
    "temperature_risk"
]

risk_data["risk_score"] = (
    risk_data[risk_columns]
    .mean(
        axis=1,
        skipna=True
    )
)


# ============================================================
# 26. RANK PATIENT-DAYS BY EXPLORATORY INDICATOR
# ============================================================

high_risk_days = (
    risk_data
    .sort_values(
        "risk_score",
        ascending=False
    )
    [
        [
            "user_id",
            "date",
            "risk_score",
            "bp_deviation",
            "heart_rate_deviation",
            "weight_deviation",
            "spo2_deviation",
            "temperature_deviation"
        ]
    ]
    .head(20)
)


print(
    "\n===== TOP 20 PATIENT-DAYS BY "
    "EXPLORATORY INDICATOR ====="
)

print(
    high_risk_days.to_string(
        index=False
    )
)


# ============================================================
# 27. SAVE FINAL RISK DATASET
# ============================================================

risk_data.to_csv(
    "patient_risk_scored.csv",
    index=False
)

print(
    "\nRisk-feature dataset saved as:"
    " patient_risk_scored.csv"
)


# ============================================================
# 28. COMPLETION MESSAGE
# ============================================================

print("\n" + "=" * 60)
print("ANALYSIS PIPELINE COMPLETED SUCCESSFULLY")
print("=" * 60)

print(
    "\nGenerated files:"
)

print("1. final_patient_day_dataset.csv")
print("2. patient_risk_features.csv")
print("3. patient_risk_scored.csv")

print(
    "\nNote: Risk indicators are exploratory and "
    "non-clinical."
)

print("=" * 60)