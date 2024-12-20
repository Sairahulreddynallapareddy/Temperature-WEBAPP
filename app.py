import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LinearRegression

# Set page configuration
st.set_page_config(page_title="Air Quality Analysis", layout="wide")

# Load data (replace with your data file path)
@st.cache_data
def load_data():
    return pd.read_csv("C:\\Users\\rakes\\Downloads\\Processed_air_data.csv")

data = load_data()

# Sidebar for navigation
st.sidebar.title("🌟 Navigation")
page = st.sidebar.radio(
    "Choose a Section:",
    ["🏠 Data Overview", "📊 EDA", "🔮 Modeling & Prediction"],
    index=0,
)

# Apply custom styles
st.markdown(
    """
    <style>
    .sidebar .sidebar-content {
        background-color: #f8f9fa;
    }
    .main-title {
        font-size: 36px;
        color: #6c63ff;
    }
    .sub-header {
        font-size: 18px;
        color: #555555;
    }
    </style>
    """,
    unsafe_allow_html=True,
)

# Data Overview Section
if page == "🏠 Data Overview":
    st.markdown("<h1 class='main-title'>Data Overview</h1>", unsafe_allow_html=True)
    st.markdown("<p class='sub-header'>Explore the structure and sample rows of your dataset.</p>", unsafe_allow_html=True)
    st.write(f"**Shape of the Dataset:** {data.shape}")
    st.write(f"**Columns:** {', '.join(data.columns)}")
    st.write("**First Few Rows:**")
    st.dataframe(data.head(), use_container_width=True)

# EDA Section
elif page == "📊 EDA":
    st.markdown("<h1 class='main-title'>Exploratory Data Analysis</h1>", unsafe_allow_html=True)
    st.markdown("<p class='sub-header'>Dive into your data to uncover hidden patterns and relationships.</p>", unsafe_allow_html=True)

    # Display heatmap
    st.write("**Correlation Heatmap:**")
    numeric_data = data.select_dtypes(include=["number"])
    if not numeric_data.empty:
        fig, ax = plt.subplots(figsize=(10, 8))
        sns.heatmap(numeric_data.corr(), annot=True, cmap="coolwarm", ax=ax, linewidths=0.5)
        st.pyplot(fig)
    else:
        st.warning("No numeric columns available for the heatmap.")

# Modeling & Prediction Section
elif page == "🔮 Modeling & Prediction":
    st.markdown("<h1 class='main-title'>Modeling & Prediction</h1>", unsafe_allow_html=True)
    st.markdown("<p class='sub-header'>Use a simple regression model to predict temperature based on input features.</p>", unsafe_allow_html=True)

    # User input form
    st.write("### Enter Feature Values:")
    features = ["PM2.5", "PM10", "SO2", "NO2", "CO", "O3"]
    inputs = []
    col1, col2, col3 = st.columns(3)
    for i, feature in enumerate(features):
        with [col1, col2, col3][i % 3]:
            value = st.number_input(f"{feature}:", value=0.0, min_value=0.0)
            inputs.append(value)

    # Simple Linear Regression Model
    if st.button("🚀 Predict Temperature"):
        model = LinearRegression()
        X = data[features].fillna(0)
        y = data["TEMP"]  # Replace with your target column
        scaler = StandardScaler()
        X_scaled = scaler.fit_transform(X)
        model.fit(X_scaled, y)

        # Scale inputs and predict
        scaled_inputs = scaler.transform([inputs])
        prediction = model.predict(scaled_inputs)
        st.success(f"🌡️ Predicted Temperature: {prediction[0]:.2f} °C")
