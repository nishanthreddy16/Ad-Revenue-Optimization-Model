import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score, mean_absolute_error, mean_squared_error
from sklearn.cluster import KMeans


# --------------------------------------------------
# PAGE CONFIGURATION
# --------------------------------------------------

st.set_page_config(
    page_title="Ad Revenue Optimization Model",
    page_icon="📊",
    layout="wide"
)


# --------------------------------------------------
# TITLE
# --------------------------------------------------

st.title("📊 Ad Revenue Optimization Model for Websites")

st.write(
    "Machine Learning model to analyze website performance "
    "and predict advertising revenue."
)


# --------------------------------------------------
# LOAD DATASET
# --------------------------------------------------

@st.cache_data
def load_data():
    df = pd.read_csv("ad_revenue_dataset.csv", on_bad_lines="skip")

    # Data cleaning
    df.fillna(0, inplace=True)
    df.drop_duplicates(inplace=True)

    # Feature Engineering
    df["engagement_score"] = (
        df["page_views"] * df["session_duration"]
    )

    df["effective_views"] = (
        df["page_views"] *
        (1 - df["bounce_rate"] / 100)
    )

    df["avg_session_per_view"] = (
        df["session_duration"] / df["page_views"]
    )

    df.replace([np.inf, -np.inf], 0, inplace=True)

    return df


df = load_data()


# --------------------------------------------------
# SIDEBAR
# --------------------------------------------------

st.sidebar.header("Navigation")

page = st.sidebar.radio(
    "Select Section",
    [
        "Dashboard",
        "Data Analysis",
        "Visualizations",
        "Revenue Prediction",
        "Clustering",
        "Model Evaluation",
        "Business Insights"
    ]
)


# --------------------------------------------------
# MODEL DEVELOPMENT
# --------------------------------------------------

X = df.drop("ad_revenue", axis=1)
y = df["ad_revenue"]

X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.2,
    random_state=42
)

model = LinearRegression()
model.fit(X_train, y_train)

y_pred = model.predict(X_test)


# --------------------------------------------------
# CLUSTERING
# --------------------------------------------------

kmeans = KMeans(
    n_clusters=3,
    random_state=42,
    n_init=10
)

df["cluster"] = kmeans.fit_predict(X)


# --------------------------------------------------
# DASHBOARD
# --------------------------------------------------

if page == "Dashboard":

    st.header("📈 Project Dashboard")

    col1, col2, col3, col4 = st.columns(4)

    with col1:
        st.metric(
            "Total Records",
            len(df)
        )

    with col2:
        st.metric(
            "Average Revenue",
            f"${df['ad_revenue'].mean():.2f}"
        )

    with col3:
        st.metric(
            "Average Page Views",
            f"{df['page_views'].mean():.0f}"
        )

    with col4:
        st.metric(
            "Average Click Rate",
            f"{df['ad_click_rate'].mean():.2f}%"
        )

    st.subheader("Dataset Preview")

    st.dataframe(
        df.head(10),
        use_container_width=True
    )

    st.subheader("Dataset Statistics")

    st.dataframe(
        df.describe(),
        use_container_width=True
    )


# --------------------------------------------------
# DATA ANALYSIS
# --------------------------------------------------

elif page == "Data Analysis":

    st.header("🔍 Data Analysis")

    st.subheader("Dataset Shape")

    col1, col2 = st.columns(2)

    with col1:
        st.metric(
            "Rows",
            df.shape[0]
        )

    with col2:
        st.metric(
            "Columns",
            df.shape[1]
        )

    st.subheader("Column Information")

    info_df = pd.DataFrame({
        "Column": df.columns,
        "Data Type": df.dtypes.astype(str),
        "Missing Values": df.isnull().sum().values
    })

    st.dataframe(
        info_df,
        use_container_width=True
    )

    st.subheader("Descriptive Statistics")

    st.dataframe(
        df.describe(),
        use_container_width=True
    )


# --------------------------------------------------
# VISUALIZATIONS
# --------------------------------------------------

elif page == "Visualizations":

    st.header("📊 Data Visualizations")

    # Revenue Distribution
    st.subheader("Revenue Distribution")

    fig, ax = plt.subplots(figsize=(8, 5))

    sns.histplot(
        df["ad_revenue"],
        bins=20,
        kde=True,
        ax=ax
    )

    ax.set_xlabel("Ad Revenue")
    ax.set_ylabel("Frequency")

    st.pyplot(fig)

    # Page Views vs Revenue
    st.subheader("Page Views vs Revenue")

    fig, ax = plt.subplots(figsize=(8, 5))

    sns.scatterplot(
        x="page_views",
        y="ad_revenue",
        data=df,
        ax=ax
    )

    ax.set_xlabel("Page Views")
    ax.set_ylabel("Ad Revenue")

    st.pyplot(fig)

    # Session Duration vs Revenue
    st.subheader("Session Duration vs Revenue")

    fig, ax = plt.subplots(figsize=(8, 5))

    sns.scatterplot(
        x="session_duration",
        y="ad_revenue",
        data=df,
        ax=ax
    )

    ax.set_xlabel("Session Duration")
    ax.set_ylabel("Ad Revenue")

    st.pyplot(fig)

    # Bounce Rate vs Revenue
    st.subheader("Bounce Rate vs Revenue")

    fig, ax = plt.subplots(figsize=(8, 5))

    sns.scatterplot(
        x="bounce_rate",
        y="ad_revenue",
        data=df,
        ax=ax
    )

    ax.set_xlabel("Bounce Rate")
    ax.set_ylabel("Ad Revenue")

    st.pyplot(fig)

    # Correlation Heatmap
    st.subheader("Correlation Heatmap")

    fig, ax = plt.subplots(figsize=(10, 7))

    sns.heatmap(
        df.drop(columns=["cluster"], errors="ignore").corr(),
        annot=True,
        ax=ax
    )

    st.pyplot(fig)


# --------------------------------------------------
# REVENUE PREDICTION
# --------------------------------------------------

elif page == "Revenue Prediction":

    st.header("💰 Ad Revenue Prediction")

    st.write(
        "Enter website performance values to predict advertising revenue."
    )

    col1, col2 = st.columns(2)

    with col1:

        page_views = st.number_input(
            "Page Views",
            min_value=0.0,
            value=10000.0
        )

        bounce_rate = st.number_input(
            "Bounce Rate (%)",
            min_value=0.0,
            max_value=100.0,
            value=40.0
        )

    with col2:

        session_duration = st.number_input(
            "Session Duration",
            min_value=0.0,
            value=300.0
        )

        ad_click_rate = st.number_input(
            "Ad Click Rate",
            min_value=0.0,
            value=3.0
        )

    if st.button("Predict Ad Revenue"):

        # Feature Engineering
        engagement_score = (
            page_views * session_duration
        )

        effective_views = (
            page_views *
            (1 - bounce_rate / 100)
        )

        avg_session_per_view = (
            session_duration / page_views
            if page_views != 0
            else 0
        )

        input_data = pd.DataFrame({
            "page_views": [page_views],
            "bounce_rate": [bounce_rate],
            "session_duration": [session_duration],
            "ad_click_rate": [ad_click_rate],
            "engagement_score": [engagement_score],
            "effective_views": [effective_views],
            "avg_session_per_view": [avg_session_per_view]
        })

        prediction = model.predict(input_data)[0]

        st.success(
            f"Predicted Ad Revenue: ${prediction:.2f}"
        )

        st.subheader("Calculated Features")

        feature_df = pd.DataFrame({
            "Feature": [
                "Engagement Score",
                "Effective Views",
                "Average Session per View"
            ],
            "Value": [
                engagement_score,
                effective_views,
                avg_session_per_view
            ]
        })

        st.dataframe(
            feature_df,
            use_container_width=True
        )


# --------------------------------------------------
# CLUSTERING
# --------------------------------------------------

elif page == "Clustering":

    st.header("🎯 Website Performance Clustering")

    st.write(
        "KMeans clustering groups websites into three performance segments."
    )

    cluster_counts = df["cluster"].value_counts().sort_index()

    st.subheader("Cluster Distribution")

    fig, ax = plt.subplots(figsize=(8, 5))

    cluster_counts.plot(
        kind="bar",
        ax=ax
    )

    ax.set_xlabel("Cluster")
    ax.set_ylabel("Number of Websites")

    st.pyplot(fig)

    st.subheader("Cluster Summary")

    cluster_summary = df.groupby("cluster").mean(
        numeric_only=True
    )

    st.dataframe(
        cluster_summary,
        use_container_width=True
    )


# --------------------------------------------------
# MODEL EVALUATION
# --------------------------------------------------

elif page == "Model Evaluation":

    st.header("🤖 Linear Regression Model Evaluation")

    r2 = r2_score(
        y_test,
        y_pred
    )

    mae = mean_absolute_error(
        y_test,
        y_pred
    )

    mse = mean_squared_error(
        y_test,
        y_pred
    )

    rmse = np.sqrt(mse)

    col1, col2, col3, col4 = st.columns(4)

    with col1:
        st.metric(
            "R² Score",
            f"{r2:.4f}"
        )

    with col2:
        st.metric(
            "MAE",
            f"{mae:.4f}"
        )

    with col3:
        st.metric(
            "MSE",
            f"{mse:.4f}"
        )

    with col4:
        st.metric(
            "RMSE",
            f"{rmse:.4f}"
        )

    st.subheader("Actual vs Predicted Revenue")

    fig, ax = plt.subplots(figsize=(8, 5))

    ax.scatter(
        y_test,
        y_pred
    )

    ax.set_xlabel("Actual Revenue")
    ax.set_ylabel("Predicted Revenue")
    ax.set_title("Actual vs Predicted Revenue")

    st.pyplot(fig)


# --------------------------------------------------
# BUSINESS INSIGHTS
# --------------------------------------------------

elif page == "Business Insights":

    st.header("💡 Business Insights")

    st.markdown("""
    ### Key Findings

    **1. Higher Page Views**
    
    Higher page views generally provide more opportunities
    for advertisements and can increase revenue.

    **2. Session Duration**
    
    Longer sessions indicate higher user engagement and
    provide more opportunities for ad interaction.

    **3. Ad Click Rate**
    
    A higher ad click rate can contribute to increased
    advertising earnings.

    **4. Bounce Rate**
    
    A lower bounce rate generally indicates better website
    engagement and performance.

    **5. Engagement**
    
    Higher engagement scores indicate that users are
    spending more time interacting with the website.

    **6. Machine Learning**
    
    Linear Regression is used to predict advertising revenue
    from website performance features.

    **7. Clustering**
    
    KMeans clustering divides websites into three
    performance groups based on their characteristics.
    """)

st.sidebar.markdown("---")
st.sidebar.info(
    "Ad Revenue Optimization Model\n\n"
    "Built using Python, Pandas, Scikit-learn, "
    "Matplotlib, Seaborn and Streamlit."
)
