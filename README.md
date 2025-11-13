---
title: "Pima Diabetes ML Project Setup and Deployment"
author: "Sidick Abdoulaye Sino"
date: "2025-11-13"
output: html_document
---

```{r setup, include=FALSE}
knitr::opts_chunk$set(echo = TRUE)
```

# 1. Project Overview

This document explains the full workflow to set up, train, and deploy the **Pima Indians Diabetes ML model** using **Streamlit** and **GitHub**.

---

# 2. GitHub Setup

1. Initialize local Git repository:

```bash
git init
```

2. Add README.md and commit:

```bash
echo "# machine_learning_lecture" >> README.md
git add README.md
git commit -m "first commit"
```

3. Add remote repository and push:

```bash
git branch -M main
git remote add origin https://github.com/sidicksino/machine_learning_lecture.git
git push -u origin main
```

4. Add project files and commit:

```bash
git add .
git commit -m "Add Streamlit app, dataset, and notebook"
git push origin main
```

---

# 3. Python Environment Setup

1. Create virtual environment:

```bash
python3 -m venv venv
```

2. Activate virtual environment:

**Mac/Linux:**

```bash
source venv/bin/activate
```

**Windows:**

```bash
venv\Scripts\activate
```

3. Install dependencies:

```bash
pip install streamlit scikit-learn pandas numpy joblib watchdog
```

4. Optional: save dependencies to `requirements.txt`:

```bash
pip freeze > requirements.txt
```

---

# 4. Training and Saving Model

```python
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
import joblib
import pandas as pd

# Load data
data = pd.read_csv('diabetes.csv')
X = data.drop('Outcome', axis=1)
y = data['Outcome']

# Scale features
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# Train model
model = LogisticRegression()
model.fit(X_scaled, y)

# Save model and scaler
joblib.dump(model, 'diabetes_model.pkl')
joblib.dump(scaler, 'scaler.pkl')
```

---

# 5. Streamlit App

Create `app.py`:

```python
import streamlit as st
import joblib
import pandas as pd
import numpy as np

# Load model and scaler
model = joblib.load('diabetes_model.pkl')
scaler = joblib.load('scaler.pkl')

st.title("Pima Diabetes Prediction")

# Input fields
pregnancies = st.number_input("Pregnancies", min_value=0)
glucose = st.number_input("Glucose")
blood_pressure = st.number_input("BloodPressure")
skin_thickness = st.number_input("SkinThickness")
insulin = st.number_input("Insulin")
bmi = st.number_input("BMI")
dpf = st.number_input("Diabetes Pedigree Function")
age = st.number_input("Age", min_value=0)

# Prepare input as DataFrame
columns = ['Pregnancies','Glucose','BloodPressure','SkinThickness','Insulin','BMI','DiabetesPedigreeFunction','Age']
input_df = pd.DataFrame([[pregnancies, glucose, blood_pressure, skin_thickness, insulin, bmi, dpf, age]], columns=columns)
input_scaled = scaler.transform(input_df)

# Prediction
prediction = model.predict(input_scaled)
probability = model.predict_proba(input_scaled)

if prediction[0] == 1:
    st.error(f"Predicted: Diabetes positive (Risk: {probability[0][1]*100:.2f}%)")
else:
    st.success(f"Predicted: Diabetes negative (Risk: {probability[0][1]*100:.2f}%)")
```

Run the app:

```bash
streamlit run app.py
```

---

# 6. Pulling the Project on a New Laptop

1. Clone the repository:

```bash
git clone https://github.com/sidicksino/machine_learning_lecture.git
cd machine_learning_lecture
```

2. Create and activate virtual environment:

```bash
python3 -m venv venv
source venv/bin/activate   # Mac/Linux
# venv\Scripts\activate   # Windows
```

3. Install dependencies:

```bash
pip install -r requirements.txt
```

4. Run the Streamlit app:

```bash
streamlit run app.py
```

5. Keep Git in sync:

```bash
git pull origin main
git add .
git commit -m "Your message"
git push origin main
```

---

# 7. Notes

* Always **retrain your model** if you upgrade `scikit-learn`.
* Pass input to `StandardScaler` as **DataFrame with same column names** to avoid warnings.
* Use `.gitignore` to avoid committing `venv/` and large datasets.
* Streamlit Cloud can be used
