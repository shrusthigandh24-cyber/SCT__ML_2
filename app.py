# -----------------------------------------------------------------
# V9.1: "The Consultant" - UI Fix for Expander Labels
# - Removed emojis from st.expander labels to fix a text rendering bug.
# -----------------------------------------------------------------

import streamlit as st
import pandas as pd
import numpy as np
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler, MinMaxScaler
import plotly.express as px
import plotly.graph_objects as go
import seaborn as sns
import matplotlib.pyplot as plt
from kneed import KneeLocator

# --- App Configuration ---
st.set_page_config(
    page_title="CustomerLens Pro | The Consultant",
    page_icon="💼",
    layout="wide"
)

# --- Initialize Session State for button toggles ---
if 'show_radar' not in st.session_state:
    st.session_state.show_radar = False
if 'show_strategy' not in st.session_state:
    st.session_state.show_strategy = False

# --- V9.0 "Corporate Clean" Theme ---
st.markdown("""
<style>
    @import url('https://fonts.googleapis.com/css2?family=Poppins:wght@400;600;700&display=swap');
    html, body, [class*="st-"], [class*="css-"] {
        font-family: 'Poppins', sans-serif;
    }
    /* Main background and text color */
    .main .block-container {
        background-color: #F0F2F6;
        color: #1E1E1E;
    }
    /* Sidebar style */
    [data-testid="stSidebar"] {
        background-color: #FFFFFF;
        border-right: 2px solid #E0E0E0;
    }
    /* Headings color */
    h1, h2, h3 {
        color: #004E9A; /* Corporate Blue */
    }
    /* Metric value color */
    [data-testid="stMetricValue"] {
        color: #0068C9;
    }
    /* Sidebar text color */
    [data-testid="stSidebar"] .st-emotion-cache-1629p8f, 
    [data-testid="stSidebar"] .st-emotion-cache-1y4p8pa, 
    [data-testid="stSidebar"] label,
    [data-testid="stSidebar"] .stSlider, 
    [data-testid="stSidebar"] .stMultiSelect, 
    [data-testid="stSidebar"] [data-testid="stNumberInput"] {
        color: #1E1E1E !important;
    }
    /* Button Style */
    .stButton>button {
        border: 2px solid #0068C9;
        background-color: #0068C9;
        color: #FFFFFF;
        padding: 0.5em 1em;
        border-radius: 0.5em;
        font-size: 1em;
        font-weight: 600;
        width: 100%;
        transition: all 0.3s ease-in-out;
    }
    .stButton>button:hover {
        transform: scale(1.02);
        background-color: #004E9A;
        border-color: #004E9A;
    }
</style>
""", unsafe_allow_html=True)


# --- Sidebar for Configuration ---
with st.sidebar:
    st.title("💼 CustomerLens Consultant")
    st.markdown("---")
    
    st.header("1. Dataset Configuration")
    num_customers = st.slider("Number of Customers", 100, 1000, 200, 50)
    age_range = st.slider("Age Range", 18, 80, (20, 70))
    income_range = st.slider("Annual Income Range (₹ Lakhs)", 1.0, 30.0, (3.0, 25.0))
    score_range = st.slider("Spending Score Range", 1, 100, (20, 90))
    random_seed = st.number_input("Random Seed", value=42)

    st.markdown("---")
    st.header("2. Clustering Configuration")
    features_to_cluster = st.multiselect(
        "Select Features for Clustering",
        options=['Age', 'Annual_Income_Lakhs', 'Spending_Score'],
        default=['Annual_Income_Lakhs', 'Spending_Score']
    )
    k_value = st.slider("Custom K Value", min_value=2, max_value=10, value=5, key="k_slider")

    st.markdown("---")
    st.header("3. Display Options")
    show_raw_data = st.checkbox("Show Raw Data Table", value=False)
    show_summary = st.checkbox("Show Statistical Summary", value=True)
    show_corr = st.checkbox("Show Correlation Matrix", value=True)
    show_elbow = st.checkbox("Show Elbow Method Plot", value=True)
    
    # --- Sidebar buttons for advanced analysis ---
    st.markdown("---")
    st.header("4. Advanced Analysis")
    st.caption("Click to show/hide advanced visuals on the main page.")

    if st.button("Cluster Comparison (Radar)"):
        st.session_state.show_radar = not st.session_state.show_radar
        
    if st.button("Marketing Strategies"):
        st.session_state.show_strategy = not st.session_state.show_strategy

# --- Helper Functions ---
@st.cache_data
def generate_data(num, age, income, score, seed):
    np.random.seed(seed)
    data = {'Age': np.random.randint(age[0], age[1] + 1, num),
            'Annual_Income_Lakhs': np.round(np.random.uniform(income[0], income[1], num), 1),
            'Spending_Score': np.random.randint(score[0], score[1] + 1, num)}
    return pd.DataFrame(data)

@st.cache_data
def create_radar_chart(df, features):
    cluster_summary = df.groupby('Cluster')[features].mean()
    scaler = MinMaxScaler()
    summary_scaled = pd.DataFrame(scaler.fit_transform(cluster_summary), index=cluster_summary.index, columns=cluster_summary.columns)
    fig = go.Figure()
    for i in summary_scaled.index:
        fig.add_trace(go.Scatterpolar(r=summary_scaled.loc[i].values, theta=summary_scaled.columns, fill='toself', name=f'Cluster {i}'))
    fig.update_layout(polar=dict(radialaxis=dict(visible=True, range=[0, 1])), showlegend=True, template='plotly_white', title="Normalized Cluster Profile Comparison")
    return fig

def get_marketing_strategy(cluster_profile, income_range, score_range):
    income, score = cluster_profile['Annual_Income_Lakhs'], cluster_profile['Spending_Score']
    high_income_threshold = income_range[0] + 0.66 * (income_range[1] - income_range[0])
    low_income_threshold = income_range[0] + 0.33 * (income_range[1] - income_range[0])
    high_score_threshold = score_range[0] + 0.66 * (score_range[1] - score_range[0])
    low_score_threshold = score_range[0] + 0.33 * (score_range[1] - score_range[0])

    if income >= high_income_threshold and score >= high_score_threshold:
        return "🏆 Champions", "**Strategy:** Target with premium products, loyalty programs, and exclusive offers. They are high-value customers worth retaining."
    elif income >= high_income_threshold and score < low_score_threshold:
        return "🤔 Cautious Affluents", "**Strategy:** Focus on quality, durability, and value-for-money. They have spending power but need convincing. Use targeted ads highlighting product benefits."
    elif income < low_income_threshold and score >= high_score_threshold:
        return "🚀 Eager Spenders", "**Strategy:** Attract with promotions, bundles, and 'buy now, pay later' options. They are enthusiastic but price-sensitive."
    elif income < low_income_threshold and score < low_score_threshold:
        return " frugal", "**Strategy:** Offer budget-friendly products, discounts, and essential items. Focus marketing on necessity and cost-saving."
    else:
        return "🎯 Mainstream", "**Strategy:** A balanced approach works best. Target with seasonal offers, new arrivals, and popular product recommendations."

# --- Data Generation ---
df = generate_data(num_customers, age_range, income_range, score_range, random_seed)

# --- Main Dashboard Display ---
st.title("Interactive Customer Segmentation Dashboard")
st.markdown("Welcome to the **CustomerLens Consultant**. This tool uses the **K-Means algorithm** to discover hidden segments in a customer dataset. Use the **Configuration Panel** on the left to control the analysis.")

st.header("📊 Data Overview & Exploration")
col1, col2, col3, col4 = st.columns(4)
col1.metric("Total Customers", f"{num_customers}")
col2.metric("Features Selected", f"{len(features_to_cluster)}")
col3.metric("Age Range", f"{age_range[0]} - {age_range[1]}")
col4.metric("Income Range (Lakhs)", f"₹{income_range[0]} - ₹{income_range[1]}L")

if show_raw_data:
    st.subheader("Raw Customer Dataset")
    st.dataframe(df)
if show_summary:
    st.subheader("Statistical Summary")
    st.dataframe(df.describe())
if show_corr and len(features_to_cluster) > 1:
    st.subheader("Feature Correlation Matrix")
    corr = df[features_to_cluster].corr()
    fig_corr, ax_corr = plt.subplots(facecolor='#F0F2F6')
    ax_corr.set_facecolor('#F0F2F6')
    sns.heatmap(corr, annot=True, cmap='Blues', ax=ax_corr, cbar_kws={'label': 'Correlation'})
    ax_corr.tick_params(colors='black')
    st.pyplot(fig_corr)

st.header("🎯 Clustering Analysis")
if len(features_to_cluster) < 2:
    st.warning("Please select at least 2 features for clustering.")
else:
    X = df[features_to_cluster]
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    if show_elbow:
        st.subheader("Elbow Method for Optimal K")
        wcss = []
        for i in range(1, 11):
            kmeans_elbow = KMeans(n_clusters=i, init='k-means++', random_state=random_seed, n_init=10).fit(X_scaled)
            wcss.append(kmeans_elbow.inertia_)
        fig_elbow, ax_elbow = plt.subplots(facecolor='#F0F2F6')
        ax_elbow.set_facecolor('#F0F2F6')
        sns.lineplot(x=range(1, 11), y=wcss, marker='o', ax=ax_elbow, color='#0068C9')
        ax_elbow.tick_params(colors='black', which='both')
        ax_elbow.set_xlabel('Number of Clusters (K)', color='black')
        ax_elbow.set_ylabel('WCSS (Distortion Score)', color='black')
        st.pyplot(fig_elbow)

    kmeans = KMeans(n_clusters=k_value, init='k-means++', random_state=random_seed, n_init=10)
    df['Cluster'] = kmeans.fit_predict(X_scaled)

    st.subheader("Cluster Visualization")
    hover_data = ['Age', 'Annual_Income_Lakhs', 'Spending_Score', 'Cluster']
    if len(features_to_cluster) == 2:
        fig_clusters = px.scatter(df, x=features_to_cluster[0], y=features_to_cluster[1], color='Cluster',
                                  color_continuous_scale='Blues', title='2D Customer Segments',
                                  hover_data=hover_data, template='plotly_white')
    elif len(features_to_cluster) == 3:
        fig_clusters = px.scatter_3d(df, x=features_to_cluster[0], y=features_to_cluster[1], z=features_to_cluster[2],
                                     color='Cluster', color_continuous_scale='Blues',
                                     title='3D Customer Segments', hover_data=hover_data, template='plotly_white')
    st.plotly_chart(fig_clusters, use_container_width=True)

    # --- CORRECTED ANALYSIS SECTION ---
    st.subheader("In-Depth Cluster Details")
    with st.expander("Cluster Profiles (Averages)"):
        cluster_summary = df.groupby('Cluster')[features_to_cluster].mean().round(2)
        st.dataframe(cluster_summary, use_container_width=True)
    with st.expander("Feature Distribution by Cluster"):
        for feature in features_to_cluster:
            fig_dist = px.box(df, x='Cluster', y=feature, color='Cluster', color_discrete_sequence=px.colors.qualitative.T10,
                              template='plotly_white', title=f'Distribution of {feature} by Cluster')
            st.plotly_chart(fig_dist, use_container_width=True)
    with st.expander("View Segmented Data"):
        st.dataframe(df, use_container_width=True)

    # --- ADVANCED ANALYSIS DISPLAY (controlled by sidebar buttons) ---
    if st.session_state.show_radar:
        st.subheader("📊 Cluster Comparison (Radar)")
        st.markdown("This radar chart visualizes the normalized profiles of each cluster, making them easy to compare.")
        radar_fig = create_radar_chart(df, features_to_cluster)
        st.plotly_chart(radar_fig, use_container_width=True)

    if st.session_state.show_strategy:
        st.subheader("💡 Marketing Strategy")
        st.markdown("Here are actionable marketing strategies tailored to each identified customer segment.")
        cluster_summary_for_strategy = df.groupby('Cluster')[['Annual_Income_Lakhs', 'Spending_Score']].mean()
        for i, row in cluster_summary_for_strategy.iterrows():
            persona, strategy = get_marketing_strategy(row, income_range, score_range)
            st.markdown(f"#### Cluster {i}: The {persona}")
            st.markdown(strategy)
            st.markdown("---")

    st.download_button(label="Download Segmented Data as CSV", data=df.to_csv(index=False).encode('utf-8'),
                        file_name=f'customer_segments_K{k_value}.csv', mime='text/csv')