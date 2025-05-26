import streamlit as st
import pandas as pd
import pickle
import json
import plotly.graph_objects as go
import plotly.express as px
import boto3
import tempfile


@st.cache_data
def load_state_county_map():
    with open("state_county_final_dict.json", 'r') as f:
        return json.load(f)

# --- Load and cache ML models from S3 ---
@st.cache_resource
def load_models(risk_columns):
    s3_bucket = 'my-model-files-ankush'
    region = 'us-east-2'

    models = {}
    s3 = boto3.client(
    's3',
    region_name=region,
    aws_access_key_id='AKIA5ZIXIO7P7FDM6TUO',
    aws_secret_access_key='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'
)


    for risk_col in risk_columns:
        model_key = f'{risk_col}_model.pkl'

        with tempfile.NamedTemporaryFile(delete=False) as tmp_file:
            s3.download_fileobj(s3_bucket, model_key, tmp_file)
            tmp_file.flush()
            with open(tmp_file.name, 'rb') as f:
                models[risk_col] = pickle.load(f)

    return models

# --- Load and cache incident data --
@st.cache_data
def load_incident_data():
    return pd.read_csv("changed_data.csv")

# --- Load data and models ---
state_county_map = load_state_county_map()
incident_df = load_incident_data()
risk_columns = ['Risk_PN_ensemble', 'Risk_NP_ensemble', 'Risk_PF_ensemble',
                'Risk_FP_ensemble', 'Risk_FN_ensemble', 'Risk_NF_ensemble']
models = load_models(risk_columns)

# --- Title ---
st.title("Risk Score Prediction for your Business")

# --- Sidebar Inputs ---
st.sidebar.header("Enter few details listed below:")
state = st.sidebar.selectbox("Select State", list(state_county_map.keys()), index=list(state_county_map.keys()).index("Virginia"))
county = st.sidebar.selectbox("Select County", state_county_map[state])
hour = st.sidebar.select_slider("Select timeframe (in hours):", options=list(range(0, 121, 12)), value=0)
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

# --- Risk Score Prediction Bar Chart ---
st.subheader(f"Risk Score Prediction for {business_state} Businesses")

user_input = pd.DataFrame({
    'name': [county],
    'state': [state],
    'Hour': [hour],
    'Business_category': [business_category],
    'incidentType': [incident_type]
})

predictions = {
    risk_col: models[risk_col].predict(user_input)[0]
    for risk_col in risk_columns
}

if business_state == 'Partial':
    transitions = {
        'Partial to Non Operational': predictions['Risk_PN_ensemble'],
        'Partial to Full Operational': predictions['Risk_PF_ensemble']
    }
    colors = ['lightblue', 'deepskyblue']
elif business_state == 'Full Operational':
    transitions = {
        'Full to Non Operational': predictions['Risk_FN_ensemble'],
        'Full to Partial Operational': predictions['Risk_FP_ensemble']
    }
    colors = ['lightgreen', 'limegreen']
else:
    transitions = {
        'Non Operational to Full Operational': predictions['Risk_NF_ensemble'],
        'Non Operational to Partial': predictions['Risk_NP_ensemble']
    }
    colors = ['lightyellow', 'gold']

labels = list(transitions.keys())
values = list(transitions.values())
max_val = max(values)
max_idx = values.index(max_val)
bar_colors = [colors[0] if i != max_idx else colors[1] for i in range(len(values))]

fig_risk = go.Figure(data=[go.Bar(
    x=labels,
    y=values,
    marker_color=bar_colors,
    text=[f"{v:.2f} pts" for v in values],
    textposition='inside',
    textfont=dict(size=20, color='black')
)])

fig_risk.update_layout(
    title=dict(text=f"Risk Score Prediction for {business_state} Businesses", font=dict(size=24, color='white')),
    xaxis=dict(title="Transition", titlefont=dict(size=18, color='white'),
               tickfont=dict(size=14, color='lightgray'), linecolor='white', mirror=True),
    yaxis=dict(title="Score (pts)", titlefont=dict(size=18, color='white'),
               tickfont=dict(size=14, color='lightgray'), linecolor='white', mirror=True),
    template="plotly_dark",
    font=dict(family="Arial", size=16, color="white")
)
st.plotly_chart(fig_risk)

# --- Pie Chart: Top 5 Counties in the selected state and incident type ---
st.subheader(f"Top 5 Counties in {state} for {incident_type}")

filtered_df = incident_df[(incident_df['state'] == state) & (incident_df['incidentType'] == incident_type)]

if filtered_df.empty:
    st.warning("Sorry, this disaster has not been presidentially declared in this state for the past 10 years.")
else:
    county_counts = filtered_df['name'].value_counts().reset_index().head(5)
    county_counts.columns = ['County', 'Incident Count']

    fig_pie = px.pie(
        county_counts,
        names='County',
        values='Incident Count',
        title=f"Top 5 Counties in {state} for {incident_type}",
        color_discrete_sequence=px.colors.sequential.RdBu
    )
    st.plotly_chart(fig_pie)

# --- Bar Chart: Top 5 Disasters in the selected state ---
st.subheader(f"Top 5 Disasters in {state}")
state_disasters = incident_df[incident_df['state'] == state]
disaster_counts = state_disasters['incidentType'].value_counts().head(5).reset_index()
disaster_counts.columns = ['Incident Type', 'Count']

fig_bar = go.Figure(data=[go.Bar(
    x=disaster_counts['Incident Type'],
    y=disaster_counts['Count'],
    marker_color='tomato',
    text=disaster_counts['Count'],
    textposition='outside'
)])
fig_bar.update_layout(
    title=f"Top 5 Disasters in {state}",
    xaxis_title="Incident Type",
    yaxis_title="Number of Occurrences",
    template="plotly_white"
)
st.plotly_chart(fig_bar)

# --- Navigation Button ---
st.markdown("---")
if st.button("More Insights on Risk Trends 📈"):
    st.switch_page("pages/Risk_Trend_Insights.py")
