# ⚡ Coal GCV Prediction AI

> **Streamlit web-app & XGBoost model for rapid *Gross Calorific Value* (GCV) prediction from coal proximate-analysis data (Air-Dried Basis).**

| Section | Info |
|---------|------|
| **Model** | XGBoost (histogram) pipeline, 5 input features |
| **Accuracy** | Test RMSE ≈ 94 kcal/kg   R² ≈ 0.947 |
| **Scope** | Ash 2-50 %, TM 3-15 %, IM 0.4-7 %, VM 5-45 %, FC 25-80 % |
| **App** | Built with Streamlit 1.33   Interactive single + batch prediction |
| **Created** | 2025-07-06   Python 3.10.13   scikit-learn 1.6.1   xgboost 2.1.4 |


---

## ✨ Features

* **Single Prediction** – enter five ADB lab values, click **Predict**, instant GCV + confidence band.  
* **Batch Processing** – upload a CSV (template provided), get predictions for hundreds of samples and download the results.  
* **Model Analytics** – global SHAP feature-importance, test metrics, reference table.  
* **Explainability** – SHAP force plot for every single prediction.  
* **Modern UI** – dark-friendly theme, emoji icons, responsive layout.

---

## 🔬 Under the Hood

| Name | Details |
|------|---------|
| **Inputs (ADB)** | `ash`, `total_moisture`, `inherent_moisture`, `volatile_matter`, `fixed_carbon` |
| **Target** | `gcv` (Gross Calorific Value, kcal / kg, Air-Dried Basis) |
| **Training rows** | 30 046 unique samples |
| **Algorithm** | `sklearn` `ColumnTransformer ➜ XGBRegressor(tree_method="hist")` |
| **Validation** | 5-fold CV hyper-search → final test split 20 % |
| **Confidence band** | ±66 kcal/kg (one standard deviation of test residuals) |

Full model card in `docs/MODEL_CARD.md`.

