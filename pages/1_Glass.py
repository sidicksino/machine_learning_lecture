import streamlit as st
import joblib
import pandas as pd
import os

st.title("Glass Type Prediction")

# Input fields
refractive_index = st.number_input("Refractive Index (RI)")
Sodium = st.number_input("Sodium (Na)")
Magnesium = st.number_input("Magnesium (Mg)")
Aluminum = st.number_input("Aluminum (Al)")
Silicon = st.number_input("Silicon (Si)")
Potassium = st.number_input("Potassium (K)")
Calcium = st.number_input("Calcium (Ca)")
Barium = st.number_input("Barium (Ba)")
Iron = st.number_input("Iron (Fe)")

column_names = ['RI', 'Na', 'Mg', 'Al', 'Si', 'K', 'Ca', 'Ba', 'Fe']

# Bouton prédiction
if st.button("Predict"):
    current_dir = os.path.dirname(__file__)
    project_root = os.path.abspath(os.path.join(current_dir, '..'))
    model_path = os.path.join(project_root, 'Glass_model.pkl')

    try:
        # Charger le modèle uniquement au clic
        model = joblib.load(model_path)
        input_data = pd.DataFrame([[refractive_index, Sodium, Magnesium, Aluminum,
                                    Silicon, Potassium, Calcium, Barium, Iron]],
                                  columns=column_names)
        prediction = model.predict(input_data)

        for type in prediction:
            if type == 1:
                st.success("Predicted: Type 1 - Building Windows Float Processed")
            elif type == 2:
                st.success("Predicted: Type 2 - Building Windows Non Float Processed")
            elif type == 3:
                st.success("Predicted: Type 3 - Vehicle Windows Float Processed")
            elif type == 5:
                st.success("Predicted: Type 5 - Containers")
            elif type == 6:
                st.success("Predicted: Type 6 - Tableware")
            elif type == 7:
                st.success("Predicted: Type 7 - Headlamps")
            else:
                st.error("Predicted: Unknown Type")

        st.dataframe(input_data, width='stretch')
    except Exception as e:
        st.error(f"Erreur lors du chargement du modèle ou prédiction : {e}")
