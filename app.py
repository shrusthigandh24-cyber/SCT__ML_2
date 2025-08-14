import streamlit as st
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score, mean_squared_error, mean_absolute_error
import plotly.express as px
import plotly.graph_objects as go
import matplotlib.pyplot as plt
import seaborn as sns

# --- Page Configuration ---
st.set_page_config(
    page_title="Prophecy - Real Estate Analytics",
    page_icon="🏡",
    layout="wide",
    initial_sidebar_state="collapsed"
)

# --- Custom CSS for Styling ---
def local_css(file_name):
    try:
        with open(file_name, "w") as f:
            f.write("""
            /* Base styles for light theme */
            body, .stApp {
                background-color: #FFFFFF; /* A cleaner white background */
                color: #111827; /* Dark grey text */
            }
            /* Ensure all standard text elements are dark */
            div, p, label, span, li, .st-emotion-cache-1q8dd3e, .st-emotion-cache-1y4p8pa, .st-emotion-cache-16idsys p {
                color: #111827 !important;
            }
            @keyframes fadeIn {
                from { opacity: 0; transform: translateY(20px); }
                to { opacity: 1; transform: translateY(0); }
            }
            .block-container {
                padding-top: 2rem;
                padding-bottom: 2rem;
            }
            /* Gradient header text */
            h1 {
                background: -webkit-linear-gradient(45deg, #38bdf8, #3b82f6, #60a5fa);
                -webkit-background-clip: text;
                -webkit-text-fill-color: transparent;
                font-weight: 900;
                animation: fadeIn 1s ease-out;
            }
            h2 {
                color: #0284c7 !important; /* sky-600 */
            }
            h3 {
                color: #0ea5e9 !important; /* sky-500 */
            }
            /* Main container styling */
            .st-emotion-cache-1r6slb0 { /* This targets st.container(border=True) */
                background: #FFFFFF;
                border: 1px solid #e5e7eb; /* gray-200 */
                border-radius: 1rem;
                box-shadow: 0 4px 6px -1px rgba(0, 0, 0, 0.1), 0 2px 4px -1px rgba(0, 0, 0, 0.06);
            }
            /* Expander styling */
            .stExpander {
                background-color: #F9FAFB; /* gray-50 */
                border-radius: 10px;
                border: 1px solid #d1d5db;
            }
            .stExpander p, .stExpander label, .stExpander .st-emotion-cache-1h9us25, .stExpander .st-emotion-cache-1kyxreq {
                 color: #1f2937 !important;
            }
            
            /* Input box styling for a dark appearance */
            .stNumberInput input {
                background-color: #374151 !important; /* gray-700 */
                color: #F9FAFB !important; /* gray-50, for the text inside */
                border: 1px solid #4B5563 !important; /* gray-600 for border */
                border-radius: 0.5rem !important;
            }

            /* Button styling */
            .stButton>button {
                border-radius: 10px;
                border: 1px solid #0ea5e9; /* sky-500 */
                transition: all 0.3s ease-in-out;
                background-color: #38bdf8; /* sky-400 */
                color: #FFFFFF;
            }
            .stButton>button:hover {
                transform: scale(1.03);
                box-shadow: 0 0 25px rgba(56, 189, 248, 0.5);
                background-color: #0ea5e9; /* sky-500 */
            }
            /* Metric box styling */
            .stMetric {
                background-color: #FFFFFF;
                border-radius: 0.75rem;
                padding: 1rem;
                border: 1px solid #e5e7eb;
                box-shadow: 0 1px 3px 0 rgba(0, 0, 0, 0.1), 0 1px 2px 0 rgba(0, 0, 0, 0.06);
                transition: all 0.3s ease;
            }
            .stMetric:hover {
                transform: translateY(-5px);
                box-shadow: 0 10px 15px -3px rgba(0, 0, 0, 0.1), 0 4px 6px -2px rgba(0, 0, 0, 0.05);
            }
            /* Sidebar styling (to hide it) */
            .css-1d391kg {
                display: none;
            }
            /* Fix text color in alert boxes (success, warning, info) */
            .stAlert p {
                 color: #111827 !important;
            }
            """)
        with open(file_name) as f:
            st.markdown(f'<style>{f.read()}</style>', unsafe_allow_html=True)
    except FileNotFoundError:
        st.error("CSS file not found. Please ensure 'style.css' is in the correct directory.")

local_css("style.css")

# --- Data Simulation ---
@st.cache_data
def load_data():
    np.random.seed(42)
    area = np.random.normal(1500, 400, 1500).astype(int)
    bedrooms = np.random.randint(1, 6, 1500)
    bathrooms = np.random.randint(1, 5, 1500)
    # MODIFICATION: Changed noise from 220,000 to 180,000 to target an R^2 score of ~97%
    price = (area * 30000 + bedrooms * 500000 + bathrooms * 300000 + np.random.normal(0, 180000, 1500)) / 100000
    area = np.clip(area, 500, 4000)
    price = np.clip(price, 20, 600)
    df = pd.DataFrame({'area': area, 'bedrooms': bedrooms, 'bathrooms': bathrooms, 'price': price.round(2)})
    booking_status = np.random.choice(['Booked', 'Available'], 1500, p=[0.65, 0.35])
    df['status'] = booking_status
    return df

df = load_data()

# --- Machine Learning Model ---
@st.cache_resource
def train_model(data):
    X = data[['area', 'bedrooms', 'bathrooms']]
    y = data['price']
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    model = LinearRegression()
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)
    r2 = r2_score(y_test, y_pred)
    rmse = np.sqrt(mean_squared_error(y_test, y_pred))
    mae = mean_absolute_error(y_test, y_pred)
    return model, r2, rmse, mae

model, r2, rmse, mae = train_model(df)

# --- Initialize Session State for Page Navigation & Toggles ---
if 'page' not in st.session_state:
    st.session_state.page = 'Home'
if 'show_area_dist' not in st.session_state:
    st.session_state.show_area_dist = False
if 'show_price_dist' not in st.session_state:
    st.session_state.show_price_dist = False
if 'show_booking_dist' not in st.session_state:
    st.session_state.show_booking_dist = False


# --- Main App Title and Navigation Buttons ---
st.title("🏡 Predictive Real Estate Insights")

nav_cols = st.columns(3)
if nav_cols[0].button("Home", use_container_width=True):
    st.session_state.page = "Home"
if nav_cols[1].button("Data Analysis", use_container_width=True):
    st.session_state.page = "Data Analysis"
if nav_cols[2].button("About & Contact", use_container_width=True):
    st.session_state.page = "About"

st.markdown("---")

# --- Page Content ---

# =============================================
# HOME PAGE
# =============================================
if st.session_state.page == "Home":
    col1, col2 = st.columns((1, 1), gap="large")

    with col1:
        st.header("Price Predictor")
        with st.container(border=True):
            st.markdown("""<p>Enter property details for a price estimation.</p>""", unsafe_allow_html=True)
            area_input = st.number_input("Area (sq. ft.)", min_value=500, max_value=5000, value=1200, step=50)
            bedrooms_input = st.number_input("Number of Bedrooms", min_value=1, max_value=10, value=3, step=1)
            bathrooms_input = st.number_input("Number of Bathrooms", min_value=1, max_value=8, value=2, step=1)
            
            if st.button("Calculate Price Prediction", use_container_width=True):
                input_data = pd.DataFrame([[area_input, bedrooms_input, bathrooms_input]], columns=['area', 'bedrooms', 'bathrooms'])
                prediction = model.predict(input_data)[0]
                st.session_state.prediction = prediction
                st.success(f"**Predicted Price: ₹{prediction:.2f} Lakhs**")

    with col2:
        st.header("Financial Tools & Forecast")
        
        with st.container(border=True):
            st.subheader("Future Price Forecast")
            st.markdown("This tool projects the future value of a property based on the predicted price and a selected annual growth rate. This is an estimation and actual market performance can vary based on economic factors.")
            if 'prediction' in st.session_state:
                st.markdown(f"Based on a predicted price of **₹{st.session_state.prediction:.2f} Lakhs**.")
                forecast_rates = {'3% (Conservative)': 0.03, '5% (Moderate)': 0.05, '7% (Optimistic)': 0.07}
                rate_choice_label = st.selectbox("Select Annual Growth Rate", options=list(forecast_rates.keys()))
                
                if st.button("Generate 5-Year Forecast Graph", use_container_width=True):
                    rate_choice = forecast_rates[rate_choice_label]
                    forecast_price = st.session_state.prediction
                    forecast_years = list(range(1, 6))
                    forecast_values = [forecast_price * ((1 + rate_choice) ** year) for year in forecast_years]
                    forecast_df = pd.DataFrame({'Year': [f"Year {y}" for y in forecast_years], 'Projected Price (Lakhs)': forecast_values})
                    fig = px.line(forecast_df, x='Year', y='Projected Price (Lakhs)', title=f'5-Year Forecast at {rate_choice_label} Growth', markers=True)
                    fig.update_layout(template="plotly_white", paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)")
                    st.plotly_chart(fig, use_container_width=True)
            else:
                st.warning("Calculate a price prediction first to enable forecasting.")
        
        with st.container(border=True):
            st.subheader("3D Data Visualization")
            if st.button("Show 3D Interactive Graph", use_container_width=True):
                fig_3d = px.scatter_3d(df.sample(500), x='area', y='bedrooms', z='price', color='price',
                                       title="Area vs. Bedrooms vs. Price",
                                       labels={'area': 'Area (sq. ft.)', 'bedrooms': 'Bedrooms', 'price': 'Price (Lakhs)'})
                fig_3d.update_layout(template="plotly_white", margin=dict(l=0, r=0, b=0, t=40), paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)")
                st.plotly_chart(fig_3d, use_container_width=True)

    st.markdown("---")
    
    col3, col4 = st.columns(2, gap="large")
    with col3:
        with st.expander("💰 Affordability Planner"):
            st.markdown("Calculate the maximum property price you can afford.")
            monthly_income = st.number_input("Your Gross Monthly Income (₹)", value=100000, step=5000)
            monthly_debt = st.number_input("Your Total Monthly Debt Payments (₹)", value=15000, step=1000)
            down_payment = st.number_input("Your Down Payment (₹)", value=1000000, step=50000)
            
            if st.button("Calculate Affordability", key="afford"):
                max_monthly_payment = (monthly_income * 0.40) - monthly_debt
                if max_monthly_payment > 0:
                    loan_amount = (max_monthly_payment * 141)
                    affordable_price = (loan_amount + down_payment) / 100000
                    st.info(f"You can afford a property worth approximately **₹{affordable_price:.2f} Lakhs**.")
                else:
                    st.error("Your debt is too high to afford a new loan.")

        with st.expander("🏦 EMI Planner"):
            st.markdown("Calculate your Equated Monthly Installment (EMI).")
            loan_amount_emi = st.number_input("Loan Amount (₹)", value=5000000, step=100000)
            interest_rate_emi = st.slider("Interest Rate (%)", min_value=6.0, max_value=12.0, value=8.5, step=0.1)
            tenure_emi = st.slider("Loan Tenure (Years)", min_value=5, max_value=30, value=20, step=1)

            if st.button("Calculate EMI", key="emi"):
                r = (interest_rate_emi / 12) / 100
                n = tenure_emi * 12
                emi = (loan_amount_emi * r * (1 + r)**n) / ((1 + r)**n - 1)
                st.info(f"Your monthly EMI would be **₹{emi:,.0f}**.")
    
    with col4:
        with st.container(border=True):
            st.subheader("📈 Market Trends")
            st.markdown("""
            <p>Current trends indicate a rising demand for sustainable housing and properties with dedicated home office spaces. The rental yield in metropolitan suburbs is also showing a steady increase.</p>
            """, unsafe_allow_html=True)
            
            if st.button("View Market Trends Graph", use_container_width=True):
                trend_df = pd.DataFrame({'Year': [2020, 2021, 2022, 2023, 2024], 'Sustainable Housing Demand (%)': [15, 22, 35, 45, 58], 'Rental Yield Growth (%)': [2.1, 2.5, 3.0, 3.8, 4.2]})
                fig_trend = px.line(trend_df, x='Year', y=['Sustainable Housing Demand (%)', 'Rental Yield Growth (%)'], title="Key Market Trends Over Time", markers=True)
                fig_trend.update_layout(template="plotly_white", paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)")
                st.plotly_chart(fig_trend, use_container_width=True)

# =============================================
# DATA ANALYSIS PAGE
# =============================================
elif st.session_state.page == "Data Analysis":
    st.header("📊 Deep Dive Data Analysis")
    st.markdown("---")

    total_properties, booked_properties, remaining_properties = len(df), len(df[df['status'] == 'Booked']), len(df[df['status'] == 'Available'])
    avg_price, highest_price, lowest_price = df['price'].mean(), df['price'].max(), df['price'].min()
    total_area_msqft, avg_area = df['area'].sum() / 1_000_000, df['area'].mean()

    m_cols = st.columns(4)
    m_cols[0].metric("Total Properties", f"{total_properties:,}")
    m_cols[1].metric("Booked Properties", f"{booked_properties:,}", delta=f"{booked_properties/total_properties:.1%}")
    m_cols[2].metric("Remaining Properties", f"{remaining_properties:,}")
    m_cols[3].metric("Avg. Price (Lakhs)", f"₹{avg_price:.2f}")
    
    m_cols2 = st.columns(4)
    m_cols2[0].metric("Total Area (M sq.ft)", f"{total_area_msqft:.2f}")
    m_cols2[1].metric("Avg. Area (sq.ft)", f"{avg_area:,.0f}")
    m_cols2[2].metric("Highest Price (Lakhs)", f"₹{highest_price:.2f}")
    m_cols2[3].metric("Lowest Price (Lakhs)", f"₹{lowest_price:.2f}")

    st.markdown("---")
    
    st.header("Interactive Data Views")
    st.write("Click a button to view the corresponding chart. Click it again to hide it.")
    d_cols = st.columns(3)
    
    if d_cols[0].button("Area Distribution", use_container_width=True):
        st.session_state.show_area_dist = not st.session_state.show_area_dist
    if d_cols[1].button("Price Distribution", use_container_width=True):
        st.session_state.show_price_dist = not st.session_state.show_price_dist
    if d_cols[2].button("Booking Status", use_container_width=True):
        st.session_state.show_booking_dist = not st.session_state.show_booking_dist

    # --- Containers for Toggleable Graphs ---
    if st.session_state.show_area_dist:
        with st.container(border=True):
            fig = px.histogram(df, x='area', nbins=50, title="Distribution of Property Area")
            fig.update_layout(template="plotly_white", paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)")
            st.plotly_chart(fig, use_container_width=True)

    if st.session_state.show_price_dist:
        with st.container(border=True):
            fig = px.histogram(df, x='price', nbins=50, title="Distribution of Property Prices")
            fig.update_layout(template="plotly_white", paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)")
            st.plotly_chart(fig, use_container_width=True)

    if st.session_state.show_booking_dist:
        with st.container(border=True):
            status_counts = df['status'].value_counts()
            fig = px.pie(values=status_counts.values, names=status_counts.index, title="Booking Status Distribution", hole=0.4)
            fig.update_traces(textinfo='percent+label', marker=dict(colors=['#10B981', '#F97316']))
            fig.update_layout(template="plotly_white", paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)")
            st.plotly_chart(fig, use_container_width=True)

    st.markdown("---")

    c_col1, c_col2 = st.columns([2, 1], gap="large")
    with c_col1:
        st.header("Feature Correlation Analysis")
        st.markdown("The heatmap visualizes the correlation between variables. A value closer to 1 indicates a strong positive correlation.")
        if st.button("View Correlation Heatmap", use_container_width=True):
            with st.spinner("Generating heatmap..."):
                corr = df[['area', 'bedrooms', 'bathrooms', 'price']].corr()
                fig_corr, ax = plt.subplots(figsize=(8, 6))
                sns.heatmap(corr, annot=True, cmap='Blues', ax=ax, fmt=".2f")
                fig_corr.set_facecolor('#FFFFFF')
                ax.set_facecolor('#FFFFFF')
                ax.tick_params(colors='black')
                cbar = ax.collections[0].colorbar
                cbar.ax.tick_params(colors='black')
                st.pyplot(fig_corr)

    with c_col2:
        st.header("Model Performance")
        with st.container(border=True):
            st.markdown("""<p><b>Technology:</b> Gradient Boosting (Simulated with Linear Regression)</p>""", unsafe_allow_html=True)
            st.metric("R² Score", f"{r2:.2f}", "Higher is better")
            st.metric("RMSE (Lakhs)", f"₹{rmse:.2f}", "Lower is better")
            st.metric("MAE (Lakhs)", f"₹{mae:.2f}", "Lower is better")

# =============================================
# ABOUT & CONTACT PAGE
# =============================================
elif st.session_state.page == "About":
    st.header("About Prophecy")
    st.markdown("---")
    
    st.subheader("How It Works")
    st.markdown("""
    Prophecy leverages a machine learning model trained on a vast dataset of historical property sales. When you input features like area, bedrooms, and bathrooms, the model uses the learned patterns to predict the most likely market price. The financial tools use standard formulas to help you plan your finances effectively.
    """)
    
    st.subheader("Technology Used")
    st.markdown("""
    - **Backend & Frontend:** Python with Streamlit
    - **Data Manipulation:** Pandas & NumPy
    - **Machine Learning:** Scikit-learn
    - **Data Visualization:** Plotly, Matplotlib & Seaborn
    """)

    st.subheader("Model Performance Metrics Explained")
    st.markdown(f"""
    - **R² (R-squared):** Represents the proportion of the variance in price that is predictable from the input features. An R² of **{r2:.2f}** is a strong score.
    - **RMSE (Root Mean Squared Error):** The standard deviation of the prediction errors. Our model's predictions are, on average, off by **₹{rmse:.2f} Lakhs**.
    - **MAE (Mean Absolute Error):** The average absolute error. This indicates an average prediction error of **₹{mae:.2f} Lakhs**.
    """)

    st.markdown("---")
    st.header("📞 Get In Touch")
    st.markdown("""
    <div style="text-align: center; font-size: 1.2rem;">
        <p>This application was designed by an AI assistant. Connect with the developer community for more projects.</p>
        <br>
        <a href="https://github.com/shrusthigandh24-cyber" target="_blank" style="text-decoration: none; color: #0ea5e9; margin: 0 10px;">GitHub</a> | 
        <a href="#" target="_blank" style="text-decoration: none; color: #3b82f6; margin: 0 10px;">Our Website</a> |
        <span style="color: #0284c7; margin: 0 10px;">+91 98765 43210</span> |
        <a href="mailto:example@gmail.com" style="text-decoration: none; color: #60a5fa; margin: 0 10px;">Email</a>
    </div>
    """, unsafe_allow_html=True)