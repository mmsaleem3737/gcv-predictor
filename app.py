import streamlit as st
import pandas as pd
import numpy as np
import joblib
import matplotlib.pyplot as plt
import shap
from datetime import datetime
import io

# Page configuration
st.set_page_config(
    page_title="Coal GCV Prediction | AI Lab",
    page_icon="⚡",
    layout="wide",
    initial_sidebar_state="expanded"
)

# --- Custom CSS (same as your original, use ... or your block here) ---
st.markdown("""<style> ... your CSS block here ... </style>""", unsafe_allow_html=True)

# Model metadata and feature info
MODEL_INFO = {
    "model_name": "xgb_gcv_5feat_v1.joblib",
    "algorithm": "XGBoost histogram-tree",
    "features": ["ash", "total_moisture", "inherent_moisture", "volatile_matter", "fixed_carbon"],
    "target": "gcv",
    "test_rmse": 94.1,
    "test_r2": 0.9470,
    "rows_trained": 30046,
    "created": "2025-07-06T13:45:00Z",
    "python": "3.10.13",
    "sklearn": "1.6.1",          # Correct version here
    "xgboost": "2.1.4"
}
FEATURE_LIST = MODEL_INFO["features"]
SIGMA = 66

FEATURE_RANGES = {
    "ash": {"min": 2.0, "max": 50.0, "direction": "↓", "description": "Higher ash lowers GCV", "icon": "🔥"},
    "total_moisture": {"min": 3.0, "max": 15.0, "direction": "↓", "description": "Higher moisture lowers GCV", "icon": "💧"},
    "inherent_moisture": {"min": 0.4, "max": 7.0, "direction": "↓", "description": "Higher inherent moisture lowers GCV", "icon": "🌊"},
    "volatile_matter": {"min": 5.0, "max": 45.0, "direction": "↑", "description": "Higher volatile matter increases GCV", "icon": "💨"},
    "fixed_carbon": {"min": 25.0, "max": 80.0, "direction": "↑", "description": "Higher fixed carbon increases GCV", "icon": "⚫"}
}
SHAP_IMPORTANCE = { "ash": 315, "inherent_moisture": 98, "fixed_carbon": 93, "total_moisture": 43, "volatile_matter": 35 }

@st.cache_resource
def load_model():
    """Load the REAL trained XGBoost pipeline"""
    MODEL_PATH = "xgb_gcv_5feat_v1.joblib"
    try:
        model = joblib.load(MODEL_PATH)
        return model
    except Exception as e:
        st.error(f"Error loading model: {e}")
        return None

@st.cache_resource
def load_explainer(_model):
    """Load SHAP explainer - cached for performance"""
    try:
        background_data = pd.DataFrame(
            np.random.uniform(0, 1, (100, 5)),
            columns=FEATURE_LIST
        )
        explainer = shap.TreeExplainer(_model.named_steps["reg"], _model.named_steps["prep"].transform(background_data))
        return explainer
    except Exception as e:
        st.error(f"Error creating SHAP explainer: {e}")
        return None

def predict_gcv(model, sample_dict):
    """Predict GCV from input features"""
    try:
        df = pd.DataFrame([sample_dict])[FEATURE_LIST]
        prediction = float(model.predict(df)[0])
        return prediction
    except Exception as e:
        st.error(f"Error making prediction: {e}")
        return None

def create_shap_force_plot(explainer, model, sample_dict):
    try:
        df = pd.DataFrame([sample_dict])[FEATURE_LIST]
        shap_values = explainer.shap_values(model.named_steps["prep"].transform(df))
        features = df.columns.tolist()
        shap_vals = shap_values[0]
        base_value = explainer.expected_value
        prediction = base_value + sum(shap_vals)

        fig, ax = plt.subplots(figsize=(12, 4))
        fig.patch.set_alpha(0.0)
        ax.patch.set_alpha(0.0)
        colors = ['#ff6b6b' if val < 0 else '#4ecdc4' for val in shap_vals]
        bars = ax.barh(features, shap_vals, color=colors, alpha=0.9)
        TEXT_COLOR = "white"
        ax.axvline(x=0, color=TEXT_COLOR, linestyle='-', linewidth=0.8, alpha=0.7)
        ax.set_xlabel('SHAP Value (Impact on Prediction)', color=TEXT_COLOR)
        ax.set_title(f'Feature Contributions\nBase Value: {base_value:.1f} → Predicted: {prediction:.1f} kcal/kg', color=TEXT_COLOR)
        ax.grid(axis='x', alpha=0.2, color=TEXT_COLOR)
        for spine in ['top', 'right', 'left', 'bottom']:
            ax.spines[spine].set_color(TEXT_COLOR)
            ax.spines[spine].set_alpha(0.5)
        ax.tick_params(axis='x', colors=TEXT_COLOR)
        ax.tick_params(axis='y', colors=TEXT_COLOR)
        for i, (bar, val) in enumerate(zip(bars, shap_vals)):
            ax.text(val, i, f'{val:.1f}',
                    va='center', ha='right' if val < 0 else 'left',
                    fontweight='bold', color='black',
                    bbox=dict(facecolor='white', alpha=0.85, pad=2, edgecolor='none'))
        plt.tight_layout()
        return fig
    except Exception as e:
        st.error(f"Error creating SHAP force plot: {e}")
        return None

def create_importance_plot():
    features = list(SHAP_IMPORTANCE.keys())
    importance_values = list(SHAP_IMPORTANCE.values())
    fig, ax = plt.subplots(figsize=(12, 6))
    fig.patch.set_alpha(0.0)
    ax.patch.set_alpha(0.0)
    colors = ['#667eea', '#764ba2', '#f093fb', '#f5576c', '#4facfe']
    bars = ax.barh(features, importance_values, color=colors, alpha=0.9)
    TEXT_COLOR = "white"
    ax.set_xlabel('Mean |SHAP| Value (kcal/kg impact units)', fontsize=12, fontweight='600', color=TEXT_COLOR)
    ax.set_title('Global Feature Importance', fontsize=14, fontweight='700', pad=20, color=TEXT_COLOR)
    ax.grid(axis='x', alpha=0.2, color=TEXT_COLOR)
    for i, (bar, value) in enumerate(zip(bars, importance_values)):
        ax.text(value + 8, i, f'{value}', va='center', fontweight='bold', fontsize=11, color=TEXT_COLOR)
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.spines['left'].set_color(TEXT_COLOR)
    ax.spines['left'].set_alpha(0.5)
    ax.spines['bottom'].set_color(TEXT_COLOR)
    ax.spines['bottom'].set_alpha(0.5)
    ax.tick_params(axis='x', colors=TEXT_COLOR)
    ax.tick_params(axis='y', colors=TEXT_COLOR)
    plt.tight_layout()
    return fig

def create_template_df():
    template_data = {
        'ash': [17.2, 25.0, 12.5, 30.0, 8.0],
        'total_moisture': [8.0, 10.5, 6.0, 12.0, 5.5],
        'inherent_moisture': [2.1, 3.0, 1.5, 4.0, 1.0],
        'volatile_matter': [31.0, 28.0, 35.0, 25.0, 38.0],
        'fixed_carbon': [49.6, 45.0, 52.0, 40.0, 55.0]
    }
    df = pd.DataFrame(template_data)
    df.index = [f'Sample_{i+1}' for i in range(len(df))]
    return df

def process_uploaded_file(uploaded_file):
    try:
        df = pd.read_csv(uploaded_file)
        required_cols = FEATURE_LIST
        missing_cols = [col for col in required_cols if col not in df.columns]
        if missing_cols:
            st.error(f"Missing required columns: {missing_cols}")
            return None
        for col in required_cols:
            feature_range = FEATURE_RANGES[col]
            invalid_rows = df[(df[col] < feature_range["min"]) | (df[col] > feature_range["max"])]
            if not invalid_rows.empty:
                st.warning(f"Warning: {len(invalid_rows)} rows have {col} values outside valid range ({feature_range['min']}-{feature_range['max']})")
        return df
    except Exception as e:
        st.error(f"Error processing file: {e}")
        return None

def main():
    st.markdown("""
    <div class="main-header">
        <h1>⚡ Coal GCV Prediction AI</h1>
        <p>Advanced machine learning model for predicting Gross Calorific Value using XGBoost</p>
    </div>
    """, unsafe_allow_html=True)
    
    model = load_model()
    if model is None:
        st.error("Failed to load model. Please check the model file.")
        return
    explainer = load_explainer(model)  # argument is _model in def, but pass as model
    
    tab1, tab2, tab3, tab4 = st.tabs(["🔮 Single Prediction", "📊 Batch Processing", "📈 Model Analytics", "ℹ️ About"])
    
    with tab1:
        col1, col2 = st.columns([1, 1], gap="large")
        with col1:
            st.markdown("### 📝 Input Coal Properties")
            st.markdown("*All values are on Air Dry Basis (ADB)*")
            inputs = {}
            for feature in FEATURE_LIST:
                feature_info = FEATURE_RANGES[feature]
                with st.container():
                    st.markdown(f"""
                    <div class="feature-card">
                        <div style="display: flex; align-items: center; gap: 10px;">
                            <span style="font-size: 1.5rem;">{feature_info['icon']}</span>
                            <div>
                                <strong>{feature.replace('_', ' ').title()}</strong>
                                <br><small>{feature_info['description']}</small>
                            </div>
                        </div>
                    </div>
                    """, unsafe_allow_html=True)
                    value = st.number_input(
                        f"",
                        min_value=feature_info["min"],
                        max_value=feature_info["max"],
                        value=(feature_info["min"] + feature_info["max"]) / 2,
                        step=0.1,
                        key=f"input_{feature}",
                        help=f"Range: {feature_info['min']}-{feature_info['max']}% | Direction: {feature_info['direction']}"
                    )
                    inputs[feature] = value
            if st.button("🔮 Predict GCV", type="primary", use_container_width=True):
                prediction = predict_gcv(model, inputs)
                if prediction is not None:
                    st.session_state.prediction = prediction
                    st.session_state.inputs = inputs.copy()
                    st.rerun()
        with col2:
            st.markdown("### 📊 Prediction Results")
            if 'prediction' in st.session_state:
                prediction = st.session_state.prediction
                st.markdown(f"""
                <div class="big-metric">
                    <h1>{prediction:.1f}</h1>
                    <p>kcal/kg</p>
                </div>
                """, unsafe_allow_html=True)
                lower_bound = prediction - SIGMA
                upper_bound = prediction + SIGMA
                col_lower, col_upper = st.columns(2)
                with col_lower:
                    st.metric("Lower Bound", f"{lower_bound:.1f}", "kcal/kg")
                with col_upper:
                    st.metric("Upper Bound", f"{upper_bound:.1f}", "kcal/kg")
                if prediction > 6000:
                    quality_class = "quality-high"
                    quality_text = "🟢 High Quality Coal"
                elif prediction > 5000:
                    quality_class = "quality-medium"
                    quality_text = "🟡 Medium Quality Coal"
                else:
                    quality_class = "quality-low"
                    quality_text = "🔴 Low Quality Coal"
                st.markdown(f"""
                <div style="text-align: center; margin-top: 1rem;">
                    <span class="{quality_class}">{quality_text}</span>
                </div>
                """, unsafe_allow_html=True)
                st.markdown("### 🔍 Feature Impact Analysis")
                if explainer is not None:
                    shap_fig = create_shap_force_plot(explainer, model, st.session_state.inputs)
                    if shap_fig is not None:
                        st.pyplot(shap_fig)
                st.markdown("### 💾 Download Result")
                download_data = {
                    "timestamp": datetime.now().isoformat(),
                    "prediction_kcal_kg": prediction,
                    "confidence_interval_lower": lower_bound,
                    "confidence_interval_upper": upper_bound,
                    **st.session_state.inputs
                }
                download_df = pd.DataFrame([download_data])
                csv_buffer = io.StringIO()
                download_df.to_csv(csv_buffer, index=False)
                csv_data = csv_buffer.getvalue()
                st.download_button(
                    label="📥 Download Results as CSV",
                    data=csv_data,
                    file_name=f"gcv_prediction_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv",
                    mime="text/csv"
                )
            else:
                st.markdown("""
                <div style="text-align: center; padding: 3rem; color: #64748b;">
                    <h3>👆 Enter coal properties and predict GCV</h3>
                    <p>Fill in the feature values on the left and click 'Predict GCV' to see results</p>
                </div>
                """, unsafe_allow_html=True)
    
    # ... your tab2, tab3, tab4 code here ...

if __name__ == "__main__":
    main()
