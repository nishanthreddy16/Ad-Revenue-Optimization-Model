import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score, mean_absolute_error, mean_squared_error


# -------------------------------------------------
# PAGE CONFIG
# -------------------------------------------------

st.set_page_config(
    page_title="Ad Revenue Optimizer",
    page_icon="📈",
    layout="wide"
)


# -------------------------------------------------
# LOAD DATA
# -------------------------------------------------

@st.cache_data
def load_data():

    df = pd.read_csv(
        "ad_revenue_dataset.csv",
        on_bad_lines="skip"
    )

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
        df["session_duration"] /
        df["page_views"].replace(0, np.nan)
    )

    df.replace(
        [np.inf, -np.inf],
        0,
        inplace=True
    )

    df.fillna(0, inplace=True)

    return df


df = load_data()


# -------------------------------------------------
# TRAIN MODEL
# -------------------------------------------------

features = [
    "page_views",
    "bounce_rate",
    "session_duration",
    "ad_click_rate",
    "engagement_score",
    "effective_views",
    "avg_session_per_view"
]

X = df[features]
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

r2 = r2_score(y_test, y_pred)
mae = mean_absolute_error(y_test, y_pred)
rmse = np.sqrt(
    mean_squared_error(y_test, y_pred)
)


# -------------------------------------------------
# SIDEBAR INPUTS
# -------------------------------------------------

st.sidebar.title("⚙️ Input Parameters")

st.sidebar.write(
    "Adjust the values to predict advertising revenue."
)

page_views = st.sidebar.slider(
    "Page Views",
    min_value=int(df["page_views"].min()),
    max_value=int(df["page_views"].max()),
    value=int(df["page_views"].median())
)

bounce_rate = st.sidebar.slider(
    "Bounce Rate (%)",
    min_value=float(df["bounce_rate"].min()),
    max_value=float(df["bounce_rate"].max()),
    value=float(df["bounce_rate"].median()),
    step=0.1
)

session_duration = st.sidebar.slider(
    "Session Duration",
    min_value=float(df["session_duration"].min()),
    max_value=float(df["session_duration"].max()),
    value=float(df["session_duration"].median()),
    step=1.0
)

ad_click_rate = st.sidebar.slider(
    "Ad Click Rate",
    min_value=float(df["ad_click_rate"].min()),
    max_value=float(df["ad_click_rate"].max()),
    value=float(df["ad_click_rate"].median()),
    step=0.1
)


# -------------------------------------------------
# CALCULATE FEATURES
# -------------------------------------------------

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


# -------------------------------------------------
# HEADER
# -------------------------------------------------

st.title("📈 Ad Revenue Optimizer")

st.write(
    "Linear Regression • Website Performance • Revenue Prediction"
)


# -------------------------------------------------
# TOP METRICS
# -------------------------------------------------

col1, col2, col3, col4 = st.columns(4)

with col1:
    st.metric(
        "Predicted Revenue",
        f"${prediction:,.2f}"
    )

with col2:
    st.metric(
        "R² Score",
        f"{r2:.4f}"
    )

with col3:
    st.metric(
        "MAE",
        f"{mae:.2f}"
    )

with col4:
    st.metric(
        "RMSE",
        f"{rmse:.2f}"
    )


st.divider()


# -------------------------------------------------
# INPUT SUMMARY + PREDICTION
# -------------------------------------------------

left, right = st.columns([1.4, 1])

with left:

    st.subheader("📋 Input Summary")

    summary = pd.DataFrame({
        "Parameter": [
            "Page Views",
            "Bounce Rate",
            "Session Duration",
            "Ad Click Rate"
        ],
        "Value": [
            f"{page_views:,}",
            f"{bounce_rate:.2f}%",
            f"{session_duration:.2f}",
            f"{ad_click_rate:.2f}%"
        ]
    })

    st.dataframe(
        summary,
        hide_index=True,
        use_container_width=True
    )


with right:

    st.subheader("💰 Predicted Ad Revenue")

    st.success(
        f"${prediction:,.2f}"
    )

    st.write(
        "Estimated advertising revenue based on "
        "the entered website performance."
    )


# -------------------------------------------------
# DERIVED FEATURES
# -------------------------------------------------

st.subheader("🔧 Derived Features")

col1, col2, col3 = st.columns(3)

with col1:
    st.metric(
        "Engagement Score",
        f"{engagement_score:,.2f}"
    )

with col2:
    st.metric(
        "Effective Views",
        f"{effective_views:,.2f}"
    )

with col3:
    st.metric(
        "Avg Session / View",
        f"{avg_session_per_view:.4f}"
    )


st.divider()


# -------------------------------------------------
# VISUALIZATIONS
# -------------------------------------------------

st.subheader("📊 Data Insights")

col1, col2 = st.columns(2)


# Revenue Distribution
with col1:

    st.write("**Revenue Distribution**")

    fig, ax = plt.subplots(
        figsize=(7, 4)
    )

    sns.histplot(
        df["ad_revenue"],
        bins=20,
        kde=True,
        ax=ax
    )

    ax.set_xlabel("Ad Revenue")
    ax.set_ylabel("Frequency")

    st.pyplot(
        fig,
        use_container_width=True
    )

    plt.close(fig)


# Page Views vs Revenue
with col2:

    st.write("**Page Views vs Revenue**")

    fig, ax = plt.subplots(
        figsize=(7, 4)
    )

    sns.scatterplot(
        data=df,
        x="page_views",
        y="ad_revenue",
        ax=ax
    )

    ax.set_xlabel("Page Views")
    ax.set_ylabel("Ad Revenue")

    st.pyplot(
        fig,
        use_container_width=True
    )

    plt.close(fig)


# -------------------------------------------------
# ACTUAL VS PREDICTED
# -------------------------------------------------

st.subheader("🎯 Actual vs Predicted Revenue")

fig, ax = plt.subplots(
    figsize=(10, 5)
)

ax.scatter(
    y_test,
    y_pred
)

# Best fit reference line
minimum = min(
    y_test.min(),
    y_pred.min()
)

maximum = max(
    y_test.max(),
    y_pred.max()
)

ax.plot(
    [minimum, maximum],
    [minimum, maximum],
    linestyle="--"
)

ax.set_xlabel("Actual Revenue")
ax.set_ylabel("Predicted Revenue")
ax.set_title("Actual vs Predicted Revenue")

st.pyplot(
    fig,
    use_container_width=True
)

plt.close(fig)


# -------------------------------------------------
# BUSINESS INSIGHT
# -------------------------------------------------

st.subheader("💡 Business Insight")

if bounce_rate < df["bounce_rate"].median():

    st.info(
        "The bounce rate is below the dataset median, "
        "which indicates relatively better user engagement."
    )

else:

    st.warning(
        "The bounce rate is relatively high. "
        "Reducing bounce rate may improve user engagement."
    )


st.caption(
    "Ad Revenue Optimization Model | "
    "Python • Pandas • Scikit-learn • Matplotlib • Seaborn • Streamlit"
)
