import streamlit as st
import pandas as pd
import joblib

st.set_page_config(page_title="Boston Housing Price Predictor", layout="centered")

# Load the trained model
@st.cache_resource
def load_model():
    return joblib.load("Real_Estates_Price_Predictor.joblib")

model = load_model()

st.title("🏡 Boston Housing Price Predictor")
st.markdown("Enter feature values to estimate the **median home price** (in $1000s).")

# Input fields
CRIM = st.number_input("CRIM: Crime rate", min_value=0.0, value=0.2)
ZN = st.number_input("ZN: % large residential zones", min_value=0.0, value=12.5)
INDUS = st.number_input("INDUS: % non-retail business", min_value=0.0, value=7.0)
CHAS = st.selectbox("CHAS: Borders Charles River?", options=[0, 1])
NOX = st.number_input("NOX: Nitric oxide conc.", min_value=0.0, max_value=1.0, value=0.5)
RM = st.number_input("RM: Avg. rooms per dwelling", min_value=1.0, max_value=10.0, value=6.0)
AGE = st.number_input("AGE: % built before 1940", min_value=0.0, max_value=100.0, value=60.0)
DIS = st.number_input("DIS: Distance to jobs", min_value=0.0, value=4.0)
RAD = st.slider("RAD: Highway access index", 1, 24, 5)
TAX = st.number_input("TAX: Property tax rate", min_value=100.0, value=300.0)
PTRATIO = st.number_input("PTRATIO: Pupil-teacher ratio", min_value=10.0, value=18.0)
B = st.number_input("B: 1000(Bk - 0.63)^2", min_value=0.0, value=300.0)
LSTAT = st.number_input("LSTAT: % lower status population", min_value=0.0, max_value=100.0, value=12.0)

# Create DataFrame for prediction
features = pd.DataFrame([[
    CRIM, ZN, INDUS, CHAS, NOX, RM, AGE,
    DIS, RAD, TAX, PTRATIO, B, LSTAT
]], columns=[
    'CRIM', 'ZN', 'INDUS', 'CHAS', 'NOX', 'RM', 'AGE',
    'DIS', 'RAD', 'TAX', 'PTRATIO', 'B', 'LSTAT'
])

# Predict button
if st.button("Predict"):
    predicted_price = model.predict(features)[0]
    st.success(f"💰 Predicted Median Home Value: **${predicted_price:.2f}k**")

st.markdown("---")
st.header("📁 Batch Prediction")

uploaded_file = st.file_uploader("Upload a CSV file with 13 features", type=["csv"])

if uploaded_file:
    try:
        batch_df = pd.read_csv(uploaded_file)

        # Check if all required columns are present
        required_columns = [
            'CRIM', 'ZN', 'INDUS', 'CHAS', 'NOX', 'RM', 'AGE',
            'DIS', 'RAD', 'TAX', 'PTRATIO', 'B', 'LSTAT'
        ]
        missing = [col for col in required_columns if col not in batch_df.columns]

        if missing:
            st.error(f"❌ Missing columns: {', '.join(missing)}")
        else:
            # Predict
            predictions = model.predict(batch_df[required_columns])
            batch_df['Predicted_MEDV'] = predictions

            st.success("✅ Predictions completed!")
            st.dataframe(batch_df.head(10))

            # Provide download link
            csv = batch_df.to_csv(index=False).encode('utf-8')
            st.download_button(
                label="📥 Download Results as CSV",
                data=csv,
                file_name="batch_predictions.csv",
                mime="text/csv"
            )
    except Exception as e:
        st.error(f"⚠️ Error reading file: {e}")


st.markdown("---")
st.markdown(
    "<small>📘 Model trained on the [Boston Housing Dataset](https://github.com/rupakc/UCI-Data-Analysis/tree/master/Boston%20Housing%20Dataset/Boston%20Housing) "
    "originally published on **July 7, 1993**, maintained by Carnegie Mellon University StatLib.</small>",
    unsafe_allow_html=True
)

st.markdown(
    "<small>© 2025 AYAN CHOWDHURY. All rights reserved.</small>",
    unsafe_allow_html=True
)
