import streamlit as st
import pandas as pd
import numpy as np
from sklearn.metrics import confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns

# ----------------------------
# Page Config
# ----------------------------
st.set_page_config(page_title="📊 Loan Policy Simulator", layout="wide")

st.title("💳 Loan Default Policy Simulator")
st.markdown("""
Adjust decision thresholds and cost assumptions to see how your policy affects business outcomes.
""")

# ----------------------------
# Load Data
# ----------------------------
df = pd.read_csv("data/scored_loans_ROS.csv")

# ----------------------------
# Sidebar Controls
# ----------------------------
st.sidebar.header("🔧 Simulation Settings")

# Threshold slider
threshold = st.sidebar.slider(
    "Decision Threshold",
    min_value=0.0, max_value=1.0,
    value=0.5, step=0.01
)

# Cost inputs
COST_FP = st.sidebar.number_input(
    "💰 Cost per False Positive (good loan wrongly rejected)",
    value=500, step=100
)
COST_FN = st.sidebar.number_input(
    "💸 Cost per False Negative (bad loan wrongly approved)",
    value=5000, step=100
)

# Filter by interest rate
if "int_rate" in df.columns:
    min_ir, max_ir = st.sidebar.slider(
        "Filter by Interest Rate (%)",
        float(df['int_rate'].min()),
        float(df['int_rate'].max()),
        (float(df['int_rate'].min()), float(df['int_rate'].max()))
    )
    df = df[(df['int_rate'] >= min_ir) & (df['int_rate'] <= max_ir)]


# Filter by DTI
if "dti" in df.columns:
    min_dti, max_dti = st.sidebar.slider(
        "Filter by DTI (Debt-to-Income Ratio)",
        float(df["dti"].min()),
        float(df["dti"].max()),
        (float(df['dti'].min()), float(df['dti'].max()))
    )
    df = df[(df['int_rate'] >= min_dti) & (df['int_rate'] <= max_dti)]

# ----------------------------
# Confusion Matrix Calculations
# ----------------------------
y_true = df['actual'].values
y_pred = (df['proba'].values >= threshold).astype(int)

tn, fp, fn, tp = confusion_matrix(y_true, y_pred).ravel()

# Costs
cost_before = 29397000.00  # from your notebook
cost_after = (fp * COST_FP) + (fn * COST_FN)
net_roi = cost_before - cost_after

# ----------------------------
# Summary Statistics
# ----------------------------
total_loans = len(df)
default_rate = df['actual'].mean()

st.subheader("📌 Summary Statistics")
colA, colB, colC = st.columns(3)
colA.metric("📦 Total Loans", f"{total_loans:,}")
colB.metric("📉 Default Rate", f"{default_rate:.2%}")
colC.metric("💵 Net Expected ROI (Savings)", f"${net_roi:,.2f}")

st.markdown("---")

# ----------------------------
# KPIs
# ----------------------------
col1, col2, col3, col4 = st.columns(4)
col1.metric("✅ True Negatives (good approved)", f"{tn:,}")
col2.metric("❌ False Positives (good rejected)", f"{fp:,}")
col3.metric("⚠️ False Negatives (bad approved)", f"{fn:,}")
col4.metric("🔎 True Positives (bad rejected)", f"{tp:,}")

col5, col6, col7, col8 = st.columns(4)
col5.metric("💸 Cost After Tuning", f"${cost_after:,.2f}")

st.markdown(f"**💡 Cost BEFORE threshold tuning:** `${cost_before:,.2f}`")
st.markdown("---")

# ----------------------------
# Confusion Matrix Plot
# ----------------------------
fig, ax = plt.subplots(figsize=(5,4))
sns.heatmap(
    [[tn, fp],[fn, tp]],
    annot=True, fmt='d', cmap='Blues',
    xticklabels=['Pred 0 (Good)', 'Pred 1 (Default)'],
    yticklabels=['Actual 0 (Good)', 'Actual 1 (Default)'],
    ax=ax
)
ax.set_title(f"Confusion Matrix (Threshold = {threshold:.2f})")
st.pyplot(fig)

# ----------------------------
# Cost vs Threshold Curve
# ----------------------------
thresholds = np.linspace(0, 1, 101)
costs = []

for t in thresholds:
    y_pred_t = (df['proba'].values >= t).astype(int)
    tn_t, fp_t, fn_t, tp_t = confusion_matrix(y_true, y_pred_t).ravel()
    costs.append((fp_t * COST_FP) + (fn_t * COST_FN))

fig2, ax2 = plt.subplots(figsize=(6,4))
ax2.plot(thresholds, costs, color='green', linewidth=2, label='Total Cost')
ax2.axvline(threshold, color='red', linestyle='--', label=f"Current Threshold ({threshold:.2f})")
ax2.set_xlabel("Decision Threshold")
ax2.set_ylabel("Total Cost ($)")
ax2.set_title("Cost vs Threshold Curve")
ax2.legend()
ax2.grid(True)
st.pyplot(fig2)

st.markdown("---")
st.markdown("✅ *Adjust the sliders in the sidebar to explore different scenarios.*")
