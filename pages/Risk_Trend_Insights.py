import streamlit as st
import pandas as pd
import pickle
import json
import plotly.express as px

# --- Load and cache state-county mapping ---
@st.cache_data
def load_state_county_map():
    with open(r"C:\Users\Lenovo\state_county_final_dict.json", 'r') as f:
        return json.load(f)

# --- Load and cache ML models ---
@st.cache_resource
def load_models(risk_columns):
    models = {}
    for risk_col in risk_columns:
        model_path = rf'C:\Users\Lenovo\{risk_col}_model.pkl'
        with open(model_path, 'rb') as file:
            models[risk_col] = pickle.load(file)
    return models

# --- Title ---
st.title("Risk Trend Insights")

# --- Sidebar Inputs ---
st.sidebar.header("Enter few details listed below:")

# --- Load data and models ---
state_county_map = load_state_county_map()
risk_columns = ['Risk_PN_ensemble', 'Risk_NP_ensemble', 'Risk_PF_ensemble',
                'Risk_FP_ensemble', 'Risk_FN_ensemble', 'Risk_NF_ensemble']
models = load_models(risk_columns)

state = st.sidebar.selectbox("Select State", list(state_county_map.keys()), index=list(state_county_map.keys()).index("Virginia"))
county = st.sidebar.selectbox("Select County", state_county_map[state])
business_category = st.sidebar.selectbox(
    "Business Category",
    ['Airports & Air Transport', 'Hospitals & Healthcare Facilities',
     'Power Plants (Electricity Generation & Distribution)',
     'Water Treatment & Utilities']
)
incident_type = st.sidebar.selectbox(
    "Incident Type",
    ['Fire', 'Tornado', 'Severe Storm', 'Hurricane', 'Flood',
     'Severe Ice Storm', 'Snowstorm', 'Mud/Landslide', 'Earthquake',
     'Coastal Storm'],
    index=6  # Default: Snowstorm
)
business_state = st.sidebar.selectbox(
    "What is the current state of your business?",
    ['Partial', 'Full Operational', 'Non Operational']
)

# --- Generate time-based predictions ---
hours = list(range(0, 121, 12))
data = pd.DataFrame({
    'Hour': hours,
    'name': [county]*len(hours),
    'state': [state]*len(hours),
    'Business_category': [business_category]*len(hours),
    'incidentType': [incident_type]*len(hours)
})

transition_col_map = {
    'Partial': ['Risk_PN_ensemble', 'Risk_PF_ensemble'],
    'Full Operational': ['Risk_FN_ensemble', 'Risk_FP_ensemble'],
    'Non Operational': ['Risk_NF_ensemble', 'Risk_NP_ensemble']
}

labels = {
    'Partial': ['Partial to Non Operational', 'Partial to Full Operational'],
    'Full Operational': ['Full to Non Operational', 'Full to Partial Operational'],
    'Non Operational': ['Non Operational to Full Operational', 'Non Operational to Partial']
}

selected_cols = transition_col_map[business_state]
selected_labels = labels[business_state]

predictions = {
    selected_labels[0]: models[selected_cols[0]].predict(data),
    selected_labels[1]: models[selected_cols[1]].predict(data)
}

# --- Plot line chart ---
st.subheader(f"How risk changes from 0 to 120 hours for {business_state} Businesses")

line_df = pd.DataFrame({
    'Hour': hours,
    selected_labels[0]: predictions[selected_labels[0]],
    selected_labels[1]: predictions[selected_labels[1]]
})

fig_line = px.line(
    line_df,
    x='Hour',
    y=[selected_labels[0], selected_labels[1]],
    labels={'value': 'Risk Score', 'Hour': 'Time (hours)', 'variable': 'Transition'},
    title="Risk Score Over Time"
)
fig_line.update_layout(template="plotly_dark")
st.plotly_chart(fig_line)
