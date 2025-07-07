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
    
    with tab2:
        st.markdown("### 📊 Batch Processing")
        st.markdown("Upload a CSV file with multiple coal samples for batch GCV prediction.")
        
        col1, col2 = st.columns([1, 1], gap="large")
        
        with col1:
            st.markdown("#### 📤 Upload Data")
            st.markdown("Upload a CSV file containing coal analysis data. Required columns:")
            for feature in FEATURE_LIST:
                feature_info = FEATURE_RANGES[feature]
                st.markdown(f"- **{feature}**: {feature_info['description']} ({feature_info['min']}-{feature_info['max']}%)")
            
            uploaded_file = st.file_uploader(
                "Choose CSV file",
                type="csv",
                help="Upload CSV with columns: " + ", ".join(FEATURE_LIST)
            )
            
            st.markdown("#### 📋 Template Data")
            st.markdown("Use this template to format your data:")
            template_df = create_template_df()
            st.dataframe(template_df, use_container_width=True)
            
            # Download template
            template_csv = io.StringIO()
            template_df.to_csv(template_csv, index=True)
            template_data = template_csv.getvalue()
            st.download_button(
                label="📥 Download Template CSV",
                data=template_data,
                file_name="coal_gcv_template.csv",
                mime="text/csv"
            )
        
        with col2:
            if uploaded_file is not None:
                df = process_uploaded_file(uploaded_file)
                if df is not None:
                    st.markdown("#### 🔍 Data Preview")
                    st.dataframe(df.head(10), use_container_width=True)
                    
                    if st.button("🚀 Process Batch", type="primary", use_container_width=True):
                        progress_bar = st.progress(0)
                        predictions = []
                        
                        for i, row in df.iterrows():
                            sample_dict = row[FEATURE_LIST].to_dict()
                            prediction = predict_gcv(model, sample_dict)
                            if prediction is not None:
                                predictions.append({
                                    'sample_id': i,
                                    'predicted_gcv': prediction,
                                    'lower_bound': prediction - SIGMA,
                                    'upper_bound': prediction + SIGMA,
                                    'quality_class': 'High' if prediction > 6000 else 'Medium' if prediction > 5000 else 'Low'
                                })
                            progress_bar.progress((i + 1) / len(df))
                        
                        if predictions:
                            results_df = pd.DataFrame(predictions)
                            original_with_predictions = df.copy()
                            original_with_predictions['predicted_gcv'] = results_df['predicted_gcv'].values
                            original_with_predictions['quality_class'] = results_df['quality_class'].values
                            original_with_predictions['lower_bound'] = results_df['lower_bound'].values
                            original_with_predictions['upper_bound'] = results_df['upper_bound'].values
                            
                            st.markdown("#### 📊 Batch Results")
                            st.dataframe(original_with_predictions, use_container_width=True)
                            
                            # Summary statistics
                            st.markdown("#### 📈 Summary Statistics")
                            col_stats1, col_stats2, col_stats3 = st.columns(3)
                            with col_stats1:
                                st.metric("Total Samples", len(results_df))
                            with col_stats2:
                                st.metric("Average GCV", f"{results_df['predicted_gcv'].mean():.1f} kcal/kg")
                            with col_stats3:
                                st.metric("GCV Range", f"{results_df['predicted_gcv'].min():.0f}-{results_df['predicted_gcv'].max():.0f}")
                            
                            # Quality distribution
                            quality_counts = results_df['quality_class'].value_counts()
                            st.markdown("#### 🏆 Quality Distribution")
                            for quality, count in quality_counts.items():
                                percentage = (count / len(results_df)) * 100
                                st.write(f"**{quality} Quality**: {count} samples ({percentage:.1f}%)")
                            
                            # Download results
                            results_csv = io.StringIO()
                            original_with_predictions.to_csv(results_csv, index=False)
                            results_data = results_csv.getvalue()
                            st.download_button(
                                label="📥 Download Batch Results",
                                data=results_data,
                                file_name=f"batch_gcv_results_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv",
                                mime="text/csv"
                            )
            else:
                st.markdown("""
                <div style="text-align: center; padding: 3rem; color: #64748b;">
                    <h3>📤 Upload CSV file to start batch processing</h3>
                    <p>Upload a CSV file with coal analysis data to predict GCV for multiple samples</p>
                </div>
                """, unsafe_allow_html=True)
    
    with tab3:
        st.markdown("### 📈 Model Analytics")
        
        col1, col2 = st.columns([1, 1], gap="large")
        
        with col1:
            st.markdown("#### 🎯 Model Performance")
            perf_col1, perf_col2 = st.columns(2)
            with perf_col1:
                st.metric("R² Score", f"{MODEL_INFO['test_r2']:.4f}")
                st.metric("Training Samples", f"{MODEL_INFO['rows_trained']:,}")
            with perf_col2:
                st.metric("RMSE", f"{MODEL_INFO['test_rmse']:.1f} kcal/kg")
                st.metric("Confidence Interval", f"±{SIGMA} kcal/kg")
            
            st.markdown("#### 🔧 Model Configuration")
            st.markdown(f"**Algorithm**: {MODEL_INFO['algorithm']}")
            st.markdown(f"**Features**: {len(MODEL_INFO['features'])} input variables")
            st.markdown(f"**Target**: {MODEL_INFO['target'].upper()}")
            st.markdown(f"**Created**: {MODEL_INFO['created'][:10]}")
            
            st.markdown("#### 🏷️ Feature Ranges")
            for feature, info in FEATURE_RANGES.items():
                st.markdown(f"**{feature.replace('_', ' ').title()}** {info['icon']}")
                st.markdown(f"- Range: {info['min']}-{info['max']}%")
                st.markdown(f"- Impact: {info['description']}")
                st.markdown("---")
        
        with col2:
            st.markdown("#### 📊 Global Feature Importance")
            importance_fig = create_importance_plot()
            if importance_fig is not None:
                st.pyplot(importance_fig)
            
            st.markdown("#### 🎯 Feature Impact Details")
            sorted_features = sorted(SHAP_IMPORTANCE.items(), key=lambda x: x[1], reverse=True)
            for i, (feature, importance) in enumerate(sorted_features, 1):
                feature_info = FEATURE_RANGES[feature]
                st.markdown(f"""
                **{i}. {feature.replace('_', ' ').title()}** {feature_info['icon']}
                - SHAP Impact: {importance} kcal/kg
                - Direction: {feature_info['direction']}
                - {feature_info['description']}
                """)
        
        st.markdown("#### 🔬 Technical Details")
        tech_col1, tech_col2, tech_col3 = st.columns(3)
        with tech_col1:
            st.markdown("**Environment**")
            st.markdown(f"- Python: {MODEL_INFO['python']}")
            st.markdown(f"- Scikit-learn: {MODEL_INFO['sklearn']}")
            st.markdown(f"- XGBoost: {MODEL_INFO['xgboost']}")
        with tech_col2:
            st.markdown("**Model Pipeline**")
            st.markdown("- Data preprocessing")
            st.markdown("- Feature scaling")
            st.markdown("- XGBoost regression")
        with tech_col3:
            st.markdown("**Validation**")
            st.markdown("- Cross-validation")
            st.markdown("- Hold-out test set")
            st.markdown("- SHAP analysis")
    
    with tab4:
        st.markdown("### ℹ️ About This Application")
        
        about_col1, about_col2 = st.columns([1, 1], gap="large")
        
        with about_col1:
            st.markdown("#### 🎯 Purpose")
            st.markdown("""
            This application predicts the Gross Calorific Value (GCV) of coal samples using advanced machine learning techniques. 
            GCV is a critical parameter that determines the energy content and commercial value of coal.
            """)
            
            st.markdown("#### 🔬 How It Works")
            st.markdown("""
            1. **Input Features**: The model uses 5 key coal properties (ash, moisture, volatile matter, fixed carbon)
            2. **Machine Learning**: XGBoost algorithm trained on 30,000+ coal samples
            3. **Prediction**: Provides GCV prediction with confidence intervals
            4. **Explainability**: SHAP values show how each feature impacts the prediction
            """)
            
            st.markdown("#### 📊 Key Features")
            st.markdown("""
            - **Single Prediction**: Predict GCV for individual coal samples
            - **Batch Processing**: Upload CSV files for multiple predictions
            - **Model Analytics**: Understand feature importance and model performance
            - **Explainable AI**: SHAP analysis for prediction interpretability
            - **Quality Classification**: Automatic coal quality assessment
            """)
        
        with about_col2:
            st.markdown("#### 🎓 Model Information")
            st.markdown(f"""
            - **Algorithm**: {MODEL_INFO['algorithm']}
            - **Performance**: R² = {MODEL_INFO['test_r2']:.4f}, RMSE = {MODEL_INFO['test_rmse']:.1f} kcal/kg
            - **Training Data**: {MODEL_INFO['rows_trained']:,} samples
            - **Features**: {len(MODEL_INFO['features'])} input variables
            - **Confidence**: ±{SIGMA} kcal/kg prediction interval
            """)
            
            st.markdown("#### 🔍 Input Requirements")
            st.markdown("All values should be on **Air Dry Basis (ADB)**:")
            for feature, info in FEATURE_RANGES.items():
                st.markdown(f"- **{feature.replace('_', ' ').title()}**: {info['min']}-{info['max']}% {info['icon']}")
            
            st.markdown("#### 🏆 Quality Classification")
            st.markdown("""
            - **High Quality**: > 6000 kcal/kg 🟢
            - **Medium Quality**: 5000-6000 kcal/kg 🟡  
            - **Low Quality**: < 5000 kcal/kg 🔴
            """)
            
            st.markdown("#### 🛠️ Technical Stack")
            st.markdown("""
            - **Backend**: Python, XGBoost, Scikit-learn
            - **Frontend**: Streamlit
            - **Visualization**: Matplotlib, SHAP
            - **Data Processing**: Pandas, NumPy
            """)
        
        st.markdown("---")
        st.markdown("#### 📝 Usage Guidelines")
        st.markdown("""
        1. **Data Quality**: Ensure input values are within specified ranges
        2. **Units**: All percentages are on Air Dry Basis (ADB)
        3. **Accuracy**: Model provides ±66 kcal/kg confidence interval
        4. **Validation**: Always validate critical decisions with laboratory analysis
        5. **Scope**: Model trained on diverse coal samples, best for similar coal types
        """)
        
        st.markdown("#### ⚠️ Disclaimers")
        st.markdown("""
        - This model is for estimation purposes and should not replace laboratory analysis for critical decisions
        - Predictions are based on statistical patterns in training data
        - Model performance may vary for coal types not well-represented in training data
        - Always validate results with certified laboratory testing when required
        """)

if __name__ == "__main__":
    main()
