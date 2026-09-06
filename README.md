# Virtual Healthcare Data Analytics

A healthcare data analytics project developed as part of my **YuvaIntern Virtual Healthcare Data Analytics Internship**.

The project focuses on analyzing remote health monitoring data from older adults with a history of heart failure and exploring data-driven approaches for identifying potential unusual health patterns.

## 🎯 Project Objective

The main objective is to build a structured healthcare analytics workflow using physiological monitoring data.

The project covers:

- Healthcare data extraction and cleaning
- Exploratory Data Analysis (EDA)
- Patient-specific baseline analysis
- Deviation and risk-pattern analysis
- Anomaly detection
- Visualization and reporting

> **Important:** This project is intended for educational and exploratory analytics. The risk indicators are not clinically validated and should not be interpreted as medical diagnoses or confirmed clinical deterioration.

## 📊 Dataset

The project uses the **PerHeart Pilot Dataset: Wrist-worn Inertial Sensor and Physiological Data from Older Adults with Heart Failure**.

The dataset contains physiological and monitoring measurements collected from older adults with a history of heart failure.

**Official Dataset:**  
https://zenodo.org/records/17459937

The dataset publication is:

**"Multimodal Dataset of In-Home Physiological and Inertial Measurements from Older Heart Failure Patients"**  
MDPI Data, 2026.

## 🛠️ Technologies Used

- Python
- Pandas
- NumPy
- Matplotlib
- Seaborn
- Scikit-learn

## 📅 Project Progress

### Week 1 – Planning & Strategic Analysis

- Defined the healthcare problem statement
- Established project objectives and hypothesis
- Selected the PerHeart healthcare dataset
- Designed the analytical strategy for remote health monitoring
- Reviewed recent research and trends in remote patient monitoring

### Week 2 – Data Extraction & Cleaning

- Loaded multiple physiological healthcare datasets using Python
- Inspected dataset structure, missing values and duplicates
- Converted UNIX timestamps into datetime format
- Checked patient IDs and measurement date ranges
- Identified and removed **73 duplicate temperature records**
- Aggregated measurements into a structured **789 patient-day dataset**
- Processed data covering **27 participants**
- Created patient-specific baseline and deviation features
- Prepared the cleaned data for further exploratory analysis

## 📈 Current Analytical Approach

The current workflow uses patient-specific baseline measurements to calculate deviations in physiological variables such as:

- Blood pressure
- Heart rate
- Weight
- SpO₂
- Temperature
- Glucose

These deviations are being explored as potential indicators of unusual health patterns.

The current risk indicator is **exploratory and relative**, rather than a clinical prediction model.

## 🚀 Upcoming Work

### Week 3 – Exploratory Data Analysis & Visualization

- Descriptive statistics
- Distribution analysis
- Patient-level variation
- Temporal trends
- Correlation analysis
- Healthcare data visualization

### Week 4 – Predictive Modeling & Anomaly Detection

- Feature selection
- Personalized anomaly detection
- Model evaluation
- Identification of unusual health patterns
- Careful interpretation of model outputs

### Week 5 – Reporting & Future Strategies

- Final analysis
- Interpretation of findings
- Project limitations
- Recommendations
- Future improvements
- Final healthcare analytics report

## 📁 Repository Structure

```text
virtual-healthcare-data-analytics/
│
├── README.md
├── analysis.py
├── Week-1-Planning-and-Strategic-Analysis.docx
└── Week2-Data-Extraction-and-Cleaning.docx
