import streamlit as st
import pandas as pd
import numpy as np
import joblib
import matplotlib.pyplot as plt
import shap
from datetime import datetime
import io

# --------------------------------------------------
#  Page & basic styling
# --------------------------------------------------
st.set_page_config(
    page_title="Coal GCV Prediction | AI Lab",
    page_icon="⚡",
    layout="wide",
    initial_sidebar_state="expanded"
)

st.markdown("""<style>
/*  put your long CSS block here (omitted for brevity) */
</style>""", unsafe_allow_html=True)

# --------------------------------------------------
#  Metadata & constants
# --------------------------------------------------
MODEL_INFO = {
    "model_name": "xgb_gcv_5feat_v1.joblib",
    "algorithm": "XGBoost histogram-tree",
    "features": ["ash", "total_moisture", "inherent_moisture",
                 "volatile_matter", "fixed_carbon"],
    "target": "gcv",
    "test_rmse": 94.1,
    "test_r2": 0.9470,
    "rows_trained": 30_046,
    "created": "2025-07-06",
    "python": "3.10.13",
    "sklearn": "1.6.1",
    "xgboost": "2.1.4"
}
FEATURE_LIST = MODEL_INFO["features"]
SIGMA = 66          # ± kcal/kg  (1 σ)

FEATURE_RANGES = {
    "ash":              dict(min=2,   max=50,  direction="↓", description="Higher ash lowers GCV", icon="🔥"),
    "total_moisture":   dict(min=3,   max=15,  direction="↓", description="Higher moisture lowers GCV", icon="💧"),
    "inherent_moisture":dict(min=0.4, max=7,   direction="↓", description="Higher inherent moisture lowers GCV", icon="🌊"),
    "volatile_matter":  dict(min=5,   max=45,  direction="↑", description="Higher volatile matter increases GCV", icon="💨"),
    "fixed_carbon":     dict(min=25,  max=80,  direction="↑", description="Higher fixed carbon increases GCV", icon="⚫")
}
SHAP_IMPORTANCE = {"ash": 315, "inherent_moisture": 98,
                   "fixed_carbon": 93, "total_moisture": 43, "volatile_matter": 35}

# --------------------------------------------------
#  Cached loaders
# --------------------------------------------------
@st.cache_resource
def load_model():
    try:
        return joblib.load(MODEL_INFO["model_name"])
    except Exception as e:
        st.error(f"❌ Could not load model: {e}")
        return None

@st.cache_resource
def load_explainer(_model):
    try:
        bg = pd.DataFrame(np.random.uniform(0, 1, (100, 5)), columns=FEATURE_LIST)
        return shap.TreeExplainer(_model.named_steps["reg"],
                                  _model.named_steps["prep"].transform(bg))
    except Exception as e:
        st.error(f"❌ Could not create SHAP explainer: {e}")
        return None

# --------------------------------------------------
#  Helper functions
# --------------------------------------------------
def predict_gcv(model, sample_dict):
    df = pd.DataFrame([sample_dict])[FEATURE_LIST]
    return float(model.predict(df)[0])

def create_shap_force_plot(explainer, model, sample_dict):
    try:
        df = pd.DataFrame([sample_dict])[FEATURE_LIST]
        sv = explainer.shap_values(model.named_steps["prep"].transform(df))[0]
        base = explainer.expected_value
        pred = base + sv.sum()

        fig, ax = plt.subplots(figsize=(10, 4))
        colors = ["#ff6b6b" if v < 0 else "#4ecdc4" for v in sv]
        ax.barh(FEATURE_LIST, sv, color=colors)
        ax.axvline(0, color="white", lw=0.8)
        ax.set_title(f"SHAP contributions (pred = {pred:,.0f} kcal/kg)")
        ax.set_xlabel("Impact (kcal/kg)")
        plt.tight_layout()
        return fig
    except Exception:
        return None

def create_importance_plot():
    fig, ax = plt.subplots(figsize=(7, 4))
    order = sorted(SHAP_IMPORTANCE, key=SHAP_IMPORTANCE.get)
    bars = ax.barh(order, [SHAP_IMPORTANCE[f] for f in order], color="#667eea")
    ax.set_xlabel("|mean SHAP| (kcal/kg)")
    ax.set_title("Global feature importance")
    for bar in bars:
        ax.text(bar.get_width()+5, bar.get_y()+0.2,
                f"{bar.get_width():.0f}", color="white")
    plt.tight_layout()
    return fig

def create_template_df():
    return pd.DataFrame(
        {
            "ash":[17.2,25,12.5,30,8],
            "total_moisture":[8,10.5,6,12,5.5],
            "inherent_moisture":[2.1,3,1.5,4,1],
            "volatile_matter":[31,28,35,25,38],
            "fixed_carbon":[49.6,45,52,40,55]
        },
        index=[f"Sample_{i+1}" for i in range(5)]
    )

def process_uploaded_file(upload):
    df = pd.read_csv(upload)
    missing = [c for c in FEATURE_LIST if c not in df.columns]
    if missing:
        st.error(f"Missing columns: {missing}")
        return None
    return df

# ==================================================
#  Streamlit UI
# ==================================================
def main():
    st.markdown(
        "<h1 style='text-align:center'>⚡ Coal GCV Prediction AI</h1>",
        unsafe_allow_html=True
    )

    model = load_model()
    if model is None:
        st.stop()
    explainer = load_explainer(model)

    tab_pred, tab_batch, tab_anal, tab_about = st.tabs(
        ["🔮 Single Prediction", "📊 Batch Processing",
         "📈 Model Analytics", "ℹ️ About"]
    )

    # ------------------------------------------------
    #  TAB 1 – Single prediction
    # ------------------------------------------------
    with tab_pred:
        col_in, col_out = st.columns([1,1])
        with col_in:
            st.markdown("#### Input coal properties (ADB)")
            user_in = {}
            for f in FEATURE_LIST:
                info = FEATURE_RANGES[f]
                user_in[f] = st.number_input(
                    f"{info['icon']} {f.replace('_',' ').title()}",
                    min_value=float(info["min"]),
                    max_value=float(info["max"]),
                    value=float((info["min"]+info["max"])/2),
                    step=0.1
                )

            if st.button("Predict GCV"):
                st.session_state.pred = predict_gcv(model, user_in)
                st.session_state.user_in = user_in

        with col_out:
            if "pred" in st.session_state:
                pred = st.session_state.pred
                st.metric("Predicted GCV", f"{pred:,.1f} kcal/kg")
                st.metric("±1 σ interval",
                          f"{pred-SIGMA:,.0f} – {pred+SIGMA:,.0f}")

                if explainer:
                    fig = create_shap_force_plot(explainer, model,
                                                 st.session_state.user_in)
                    if fig: st.pyplot(fig)

    # ------------------------------------------------
    #  TAB 2 – Batch processing
    # ------------------------------------------------
    with tab_batch:
        st.markdown("### 📥 Download CSV template")
        st.download_button("Download template",
                           create_template_df().to_csv(index=False),
                           file_name="gcv_template.csv")

        st.markdown("### 📤 Upload your CSV file")
        up = st.file_uploader("Choose CSV", type="csv")
        if up:
            df_up = process_uploaded_file(up)
            if df_up is not None:
                st.dataframe(df_up.head())
                if st.button("Run predictions on file"):
                    preds = model.predict(df_up[FEATURE_LIST])
                    df_up["predicted_gcv"] = preds
                    df_up["ci_low"] = preds - SIGMA
                    df_up["ci_high"] = preds + SIGMA
                    st.success("Done!")
                    st.dataframe(df_up)
                    st.download_button(
                        "Download results",
                        df_up.to_csv(index=False),
                        file_name=f"gcv_results_{datetime.now():%Y%m%d_%H%M%S}.csv"
                    )

    # ------------------------------------------------
    #  TAB 3 – Analytics
    # ------------------------------------------------
    with tab_anal:
        c1, c2, c3 = st.columns(3)
        c1.metric("Test RMSE", f"{MODEL_INFO['test_rmse']} kcal/kg")
        c2.metric("Test R²", f"{MODEL_INFO['test_r2']:.3f}")
        c3.metric("Training rows", f"{MODEL_INFO['rows_trained']:,}")

        st.pyplot(create_importance_plot())

        # feature table
        feats = []
        for f, d in FEATURE_RANGES.items():
            feats.append({
                "Feature": f,
                "Range": f"{d['min']} – {d['max']} %",
                "Effect": d["direction"],
                "Description": d["description"]
            })
        st.dataframe(pd.DataFrame(feats))

    # ------------------------------------------------
    #  TAB 4 – About
    # ------------------------------------------------
    with tab_about:
        st.markdown("#### Model details")
        st.write(f"- **Algorithm**: {MODEL_INFO['algorithm']}")
        st.write(f"- **Features**: {', '.join(FEATURE_LIST)} (ADB)")
        st.write(f"- **Created**: {MODEL_INFO['created']}")
        st.write(f"- **Python / sklearn / xgboost**: "
                 f"{MODEL_INFO['python']} / {MODEL_INFO['sklearn']} / "
                 f"{MODEL_INFO['xgboost']}")

        st.markdown("#### How to use")
        st.write("1. Go to *Single Prediction*, enter ADB proximate values, click **Predict**.")
        st.write("2. For many samples, download the template in *Batch Processing*, "
                 "fill your rows, re-upload, then download the results CSV.")

        st.markdown("#### Limitations")
        st.write("- Model is valid for ash 2-50 %, TM 3-15 %, etc. Outside this range accuracy is unknown.")
        st.write("- Output is **Gross Calorific Value (ADB)**; convert yourself for ARB/DB if needed.")

# ==================================================
if __name__ == "__main__":
    main()
