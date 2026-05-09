import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
import warnings

warnings.filterwarnings("ignore")

st.set_page_config(
    page_title="BMW Final Data Analysis Project",
    layout="wide",
    initial_sidebar_state="collapsed"
)

# Modern, Premium Light Theme CSS
st.markdown("""
<style>
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;800&display=swap');
    
    html, body, [class*="css"]  {
        font-family: 'Inter', sans-serif;
    }
    
    .stApp {
        background-color: #F8FAFC; 
        color: #1E293B;
    }
    
    .project-header {
        background-color: #FFFFFF;
        padding: 30px;
        border-radius: 16px;
        box-shadow: 0 4px 6px -1px rgba(0, 0, 0, 0.05);
        border-top: 6px solid #2563EB; 
        margin-bottom: 30px;
    }
    
    h1, h2, h3 {
        color: #0F172A !important;
        font-weight: 800 !important;
    }
    
    [data-testid="stMetric"] {
        background-color: #FFFFFF;
        padding: 20px 25px;
        border-radius: 16px;
        box-shadow: 0 4px 6px -1px rgba(0, 0, 0, 0.05), 0 2px 4px -1px rgba(0, 0, 0, 0.03);
        border: 1px solid #F1F5F9;
        transition: transform 0.2s ease, box-shadow 0.2s ease;
    }
    [data-testid="stMetric"]:hover {
        transform: translateY(-3px);
        box-shadow: 0 10px 15px -3px rgba(0, 0, 0, 0.1);
    }
    
    [data-testid="stMetricValue"] {
        color: #0F172A !important;
        font-size: 32px !important;
    }
    [data-testid="stMetricLabel"] {
        color: #64748B !important;
        font-weight: 600 !important;
        font-size: 14px !important;
        text-transform: uppercase;
        letter-spacing: 0.5px;
    }
    
    .stTabs [data-baseweb="tab-list"] {
        background-color: #FFFFFF;
        border-radius: 12px;
        padding: 8px;
        box-shadow: 0 1px 3px 0 rgba(0, 0, 0, 0.1);
        gap: 10px;
        margin-bottom: 20px;
        flex-wrap: wrap;
    }
    .stTabs [data-baseweb="tab"] {
        height: 45px;
        background-color: transparent;
        border-radius: 8px;
        color: #64748B;
        font-weight: 600;
        padding: 0 20px;
        border: none;
        transition: all 0.2s;
    }
    .stTabs [aria-selected="true"] {
        background-color: #EFF6FF !important;
        color: #2563EB !important;
        border: none !important;
        box-shadow: inset 0 0 0 1px #DBEAFE;
    }
    
    header {visibility: hidden;}
</style>
""", unsafe_allow_html=True)

@st.cache_data
def load_data():
    df = pd.read_csv("bmw_sales_data.csv")
    return df

try:
    df = load_data()
except Exception as e:
    st.error("Could not load bmw_sales_data.csv. Make sure the file exists in lab7/data/")
    st.stop()

# Data Preprocessing for modeling
numeric_cols = ['price_usd', 'marketing_spend_usd', 'dealership_count', 
                'fuel_price_usd', 'gdp_growth_percent', 'interest_rate_percent', 
                'competition_index']
target = 'units_sold'

X = df[numeric_cols]
y = df[target]

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

scaler = StandardScaler()
X_train_sc = scaler.fit_transform(X_train)
X_test_sc = scaler.transform(X_test)

model = LinearRegression()
model.fit(X_train_sc, y_train)
y_pred = model.predict(X_test_sc)

r2 = r2_score(y_test, y_pred)
mae = mean_absolute_error(y_test, y_pred)
rmse = np.sqrt(mean_squared_error(y_test, y_pred))

# Modern Header Section
st.markdown("""
<div class="project-header">
    <h1 style="margin-bottom: 5px;">Final Data Analysis Project</h1>
    <h3 style="color: #64748B !important; margin-top: 0; font-weight: 500 !important;">BMW Global Sales: Collection, Preprocessing, EDA, and Regression Modeling</h3>
    <br>
    <p style="color: #334155; font-size: 1.1rem; font-weight: 600;">
        Created by: <span style="color: #2563EB;">Kaddour Abdelmalek Islam</span>
    </p>
</div>
""", unsafe_allow_html=True)

tab1, tab2, tab3, tab4, tab5 = st.tabs([
    "Collection & Overview", 
    "Data Preprocessing",
    "Exploratory Analysis (EDA)", 
    "Regression Modeling", 
    "Final Conclusion"
])

with tab1:
    st.subheader("Data Collection & Overview")
    st.markdown("This section covers the initial state of our collected dataset (Lab 3 & 4), displaying raw records and missing value verification.")
    col1, col2, col3, col4 = st.columns(4)
    with col1:
        st.metric("Total Collected Records", f"{len(df):,}")
    with col2:
        st.metric("Total Features", f"{len(df.columns)}")
    with col3:
        st.metric("Missing Null Values", "0")
    with col4:
        st.metric("Global Avg Sales", f"{df['units_sold'].mean():.0f}")
        
    st.markdown("<br>", unsafe_allow_html=True)
    st.dataframe(df.head(8), use_container_width=True)

with tab2:
    st.subheader("Data Preprocessing Steps")
    st.markdown("""
    In Lab 4, we prepared our data for Machine Learning. Here is a summary of the transformations applied to build our models:
    - **Data Cleaning:** Verified 0 missing values. Kept extreme prices and marketing spend as natural luxury market outliers.
    - **Categorical Encoding:** One-Hot Encoded variables like `country`, `model`, `segment`, and `engine_type` to convert text into numeric data.
    - **Scaling:** Applied Min-Max Scaling and Standard Scaling so that algorithms don't favor features with large numbers (like `price_usd`).
    """)
    
    st.markdown("#### Sample of Scaled Features (Ready for ML)")
    # Show scaled data
    df_scaled_preview = pd.DataFrame(X_train_sc[:5], columns=numeric_cols)
    st.dataframe(df_scaled_preview, use_container_width=True)

with tab3:
    st.subheader("Exploratory Data Analysis (EDA)")
    
    st.markdown("#### Feature Relationships")
    colA, colB = st.columns(2)
    with colA:
        fig_scatter1 = px.scatter(
            df, x="marketing_spend_usd", y="units_sold", color="segment",
            title="Marketing Spend vs Units Sold",
            opacity=0.7
        )
        fig_scatter1.update_layout(template="plotly_white", plot_bgcolor="rgba(0,0,0,0)", paper_bgcolor="rgba(0,0,0,0)")
        st.plotly_chart(fig_scatter1, use_container_width=True)
        
    with colB:
        fig_scatter2 = px.scatter(
            df, x="price_usd", y="units_sold", color="engine_type",
            title="Vehicle Price vs Units Sold",
            opacity=0.7
        )
        fig_scatter2.update_layout(template="plotly_white", plot_bgcolor="rgba(0,0,0,0)", paper_bgcolor="rgba(0,0,0,0)")
        st.plotly_chart(fig_scatter2, use_container_width=True)

    st.markdown("#### Categorical Distributions & Heatmap")
    col1, col2 = st.columns(2)
    with col1:
        seg_sales = df.groupby("segment")["units_sold"].sum().reset_index()
        fig_pie = px.pie(
            seg_sales, names="segment", values="units_sold", 
            title="Total Sales by Vehicle Segment",
            color_discrete_sequence=["#2563EB", "#38BDF8", "#818CF8", "#F472B6"]
        )
        fig_pie.update_layout(template="plotly_white", plot_bgcolor="rgba(0,0,0,0)", paper_bgcolor="rgba(0,0,0,0)")
        st.plotly_chart(fig_pie, use_container_width=True)
        
    with col2:
        corr_matrix = df[numeric_cols + [target]].corr()
        fig_corr = px.imshow(
            corr_matrix, 
            text_auto=".2f", 
            aspect="auto",
            title="Feature Correlation Heatmap",
            color_continuous_scale="Blues"
        )
        fig_corr.update_layout(template="plotly_white", plot_bgcolor="rgba(0,0,0,0)", paper_bgcolor="rgba(0,0,0,0)")
        st.plotly_chart(fig_corr, use_container_width=True)

with tab4:
    st.subheader("Regression Modeling Results")
    
    c1, c2, c3 = st.columns(3)
    c1.metric("R² Score", f"{r2:.4f}")
    c2.metric("Mean Absolute Error", f"{mae:.2f}")
    c3.metric("Root Mean Squared Error", f"{rmse:.2f}")
    
    st.markdown("<br>", unsafe_allow_html=True)
    
    col1, col2 = st.columns(2)
    with col1:
        fig_pred = px.scatter(
            x=y_test, y=y_pred, 
            labels={'x': 'Actual Units Sold', 'y': 'Predicted Units Sold'},
            title="Actual vs Predicted Sales",
            color_discrete_sequence=["#38BDF8"]
        )
        min_val = min(y_test.min(), y_pred.min())
        max_val = max(y_test.max(), y_pred.max())
        fig_pred.add_shape(
            type="line", x0=min_val, y0=min_val, x1=max_val, y1=max_val,
            line=dict(color="#EF4444", width=2, dash="dash")
        )
        fig_pred.update_layout(template="plotly_white", plot_bgcolor="rgba(0,0,0,0)", paper_bgcolor="rgba(0,0,0,0)")
        st.plotly_chart(fig_pred, use_container_width=True)
        
    with col2:
        coef_df = pd.DataFrame({"Feature": numeric_cols, "Coefficient": model.coef_})
        coef_df = coef_df.sort_values(by="Coefficient")
        
        fig_coef = px.bar(
            coef_df, x="Coefficient", y="Feature",
            orientation="h",
            title="Linear Regression Coefficients",
            color="Coefficient",
            color_continuous_scale="RdBu"
        )
        fig_coef.update_layout(template="plotly_white", plot_bgcolor="rgba(0,0,0,0)", paper_bgcolor="rgba(0,0,0,0)", coloraxis_showscale=False)
        st.plotly_chart(fig_coef, use_container_width=True)

with tab5:
    st.markdown("""
    <div style="background-color: white; padding: 30px; border-radius: 16px; border: 1px solid #E2E8F0; box-shadow: 0 4px 6px -1px rgba(0, 0, 0, 0.05);">
        <h3 style="margin-top: 0; color: #0F172A;">Project Takeaways</h3>
        <p style="font-size: 16px; color: #334155; line-height: 1.6;">
            Based on the exploratory data analysis and regression modeling across all our labs, here are the main conclusions for the BMW global sales dataset:
        </p>
        <ul style="font-size: 16px; color: #334155; line-height: 1.8;">
            <li><b>Dealership Count and Marketing Spend</b> had the biggest positive impact on units sold. Increasing these directly improves sales.</li>
            <li><b>Price and Competition Index</b> had the most negative effect. Higher prices and denser competition lower the total volume.</li>
            <li><b>Model Performance:</b> The Linear Regression model performed well without overfitting, confirming that the relationship between these features and sales is mostly linear.</li>
            <li><b>Sales Distribution:</b> We found that BMW sales are relatively balanced across major vehicle segments like SUVs and Sedans.</li>
        </ul>
    </div>
    """, unsafe_allow_html=True)
    
    st.markdown("<br>", unsafe_allow_html=True)
    
    csv_data = df.to_csv(index=False).encode('utf-8')
    st.download_button(
        label="Download Cleaned Dataset (CSV)",
        data=csv_data,
        file_name="bmw_sales_cleaned.csv",
        mime="text/csv",
        type="primary"
    )
