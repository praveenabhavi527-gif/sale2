import streamlit as st
import pandas as pd
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score

# -----------------------------------------------------------------------------
# 1. PAGE CONFIGURATION & STYLING
# -----------------------------------------------------------------------------
st.set_page_config(
    page_title="Purchase Prediction AI",
    page_icon="🛍️",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS for a "Very Design" look
st.markdown("""
    <style>
    .main {
        background-color: #f8f9fa;
    }
    .stButton>button {
        width: 100%;
        background-color: #ff4b4b;
        color: white;
        border-radius: 8px;
        height: 50px;
        font-weight: bold;
        border: none;
    }
    .stButton>button:hover {
        background-color: #e04444;
    }
    .metric-card {
        background-color: white;
        padding: 20px;
        border-radius: 10px;
        box-shadow: 0 4px 6px rgba(0,0,0,0.1);
        text-align: center;
    }
    h1, h2, h3 {
        color: #2c3e50;
    }
    </style>
""", unsafe_allow_html=True)

# -----------------------------------------------------------------------------
# 2. DATA HANDLING
# -----------------------------------------------------------------------------
@st.cache_data
def load_data(uploaded_file=None):
    if uploaded_file is not None:
        df = pd.read_csv(uploaded_file)
    else:
        # SYNTHETIC DATA GENERATION (Based on your Screenshot Statistics)
        # If no file is uploaded, we create data so the app actually works
        np.random.seed(42)
        rows = 200
        data = {
            'Age': np.random.randint(18, 60, rows),
            'Income': np.random.randint(15000, 90000, rows),
            'Time_On_App': np.random.normal(12, 12, rows).clip(2, 45), # Mean 12, range 2-45
            'Discount_Availed': np.random.choice([0, 1], rows),
            'Clicks': np.random.randint(1, 30, rows),
        }
        df = pd.DataFrame(data)
        # Create a target variable logic (simulated) for training
        # Logic: Higher income + More time + Discount = Higher chance
        df['Purchased'] = np.where(
            (df['Income'] > 40000) & 
            (df['Time_On_App'] > 10) & 
            ((df['Discount_Availed'] == 1) | (df['Clicks'] > 10)), 
            1, 0
        )
    return df

# -----------------------------------------------------------------------------
# 3. SIDEBAR - INPUTS
# -----------------------------------------------------------------------------
with st.sidebar:
    st.title("🛍️ User Parameters")
    st.write("Adjust the values below to predict purchase behavior.")
    
    st.divider()
    
    input_age = st.number_input("Age", min_value=18, max_value=100, value=25)
    input_income = st.number_input("Income ($)", min_value=0, value=60000, step=1000)
    input_time = st.slider("Time On App (min)", 0.0, 60.0, 20.77)
    input_discount = st.selectbox("Discount Availed?", options=[0, 1], format_func=lambda x: "Yes" if x == 1 else "No")
    input_clicks = st.slider("Number of Clicks", 0, 50, 30)
    
    st.divider()
    st.caption("Based on Logistic Regression Model")

# -----------------------------------------------------------------------------
# 4. MAIN APP LOGIC
# -----------------------------------------------------------------------------

# Header
col1, col2 = st.columns([2, 1])
with col1:
    st.title("Online Purchase Prediction")
    st.markdown("### Will the customer buy the product?")
with col2:
    st.image("https://cdn-icons-png.flaticon.com/512/3081/3081559.png", width=100)

st.divider()

# Data Loading & Training Section
# We try to load data, or ask for upload
uploaded_file = st.file_uploader("Upload CSV (Optional)", type=["csv"])
df = load_data(uploaded_file)

if df is not None:
    # Feature Selection based on your code
    features = ['Age', 'Income', 'Time_On_App', 'Discount_Availed', 'Clicks']
    target = 'Purchased'

    # Check if columns exist
    if all(col in df.columns for col in features + [target]):
        X = df[features]
        y = df[target]

        # Model Training
        # Using max_iter=1000 as per your code
        model = LogisticRegression(max_iter=1000)
        model.fit(X, y)
        
        # Calculate Accuracy for display
        score = model.score(X, y)

        # ---------------------------------------------------------------------
        # PREDICTION SECTION
        # ---------------------------------------------------------------------
        
        # Prepare input array
        input_data = [[input_age, input_income, input_time, input_discount, input_clicks]]
        prediction = model.predict(input_data)
        probability = model.predict_proba(input_data)

        # Layout for results
        res_col1, res_col2 = st.columns(2)

        with res_col1:
            st.subheader("Prediction Result")
            if prediction[0] == 1:
                st.success("## ✅ PURCHASE LIKELY")
                st.markdown(f"**Confidence:** {probability[0][1]*100:.2f}%")
            else:
                st.error("## ❌ PURCHASE UNLIKELY")
                st.markdown(f"**Confidence:** {probability[0][0]*100:.2f}%")

        with res_col2:
            st.subheader("Model Stats")
            st.info(f"Model Accuracy on current data: **{score*100:.2f}%**")
            with st.expander("View Training Data Sample"):
                st.dataframe(df.head(5))

    else:
        st.error(f"The CSV must contain the following columns: {features} and {target}")
