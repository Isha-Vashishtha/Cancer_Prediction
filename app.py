import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import joblib
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import confusion_matrix, classification_report, roc_curve, auc

def load_data():
    """Load and preprocess the dataset."""
    data = pd.read_csv('cancer.csv')
    data_cleaned = data.drop(["id", "Unnamed: 32"], axis=1)
    le = LabelEncoder()
    data_cleaned['diagnosis'] = le.fit_transform(data_cleaned['diagnosis'])
    return data_cleaned

def load_model():
    """Load the trained model and feature names."""
    return joblib.load('model.pkl'), joblib.load('features.pkl')

def main():
    st.set_page_config(page_title="Cancer Detection Assistant", layout="wide")
    
    # Custom CSS
    st.markdown("""
        <style>
        body { background-color: #f0f2f6; }
        .stButton>button { background-color: #ff4b4b; color: white; font-size: 18px; }
        .stButton>button:hover { background-color: #e63946; }
        .stTabs [data-baseweb="tab"] { font-size: 16px; font-weight: bold; }
        </style>
    """, unsafe_allow_html=True)
    
    st.title("🏥 AI-Powered Cancer Detection: Your Smart Health Companion")
    
    try:
        data_cleaned = load_data()
        model, features = load_model()
    except:
        st.error("Model or dataset not found. Upload 'cancer.csv' and trained model.")
        return
    
    X = data_cleaned.drop('diagnosis', axis=1)
    user_inputs = {}
    
    # Sidebar for Input Features
    st.sidebar.header("📝 Enter Measurements")
    for feature in features:
        user_inputs[feature] = st.sidebar.number_input(f"{feature.replace('_', ' ').title()}", value=float(X[feature].mean()))
    
    # Feature Analysis
    st.subheader("📊 Feature Analysis")
    selected_feature = st.selectbox("Select a Feature to Visualize", features)
    fig, ax = plt.subplots(1, 2, figsize=(12, 4))
    sns.histplot(data_cleaned[selected_feature], kde=True, ax=ax[0], color='blue')
    ax[0].set_title(f"Distribution of {selected_feature}")
    sns.boxplot(x=data_cleaned['diagnosis'], y=data_cleaned[selected_feature], ax=ax[1])
    ax[1].set_title(f"{selected_feature} vs Diagnosis")
    st.pyplot(fig)
    
    # Model Performance
    st.subheader("📈 Model Performance")
    y_true = data_cleaned['diagnosis']
    y_pred = model.predict(X)
    cm = confusion_matrix(y_true, y_pred)
    fig, ax = plt.subplots(figsize=(3, 2))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=['Benign', 'Malignant'], yticklabels=['Benign', 'Malignant'])
    plt.xlabel("Predicted")
    plt.ylabel("Actual")
    plt.title("Confusion Matrix")
    st.pyplot(fig)
    
    # ROC Curve
    y_prob = model.predict_proba(X)[:, 1]
    fpr, tpr, _ = roc_curve(y_true, y_prob)
    roc_auc = auc(fpr, tpr)
    fig, ax = plt.subplots()
    ax.plot(fpr, tpr, label=f'ROC Curve (area = {roc_auc:.2f})')
    ax.plot([0, 1], [0, 1], linestyle='--')
    plt.xlabel("False Positive Rate")
    plt.ylabel("True Positive Rate")
    plt.title("ROC Curve")
    plt.legend()
    st.pyplot(fig)
    
    # Prediction
    if st.button("🔍 Get Results"):
        input_vector = np.array([user_inputs[feature] for feature in features]).reshape(1, -1)
        prediction = model.predict(input_vector)[0]
        confidence_score = np.max(model.predict_proba(input_vector)) * 100
        
        st.markdown("---")
        if prediction == 0:
            st.success(f"### 🟢 Likely Benign\nConfidence: {confidence_score:.2f}%")
        else:
            st.error(f"### 🔴 Likely Malignant\nConfidence: {confidence_score:.2f}%")
        
        st.warning("⚠️ **Important:** This is a screening tool. Please consult a doctor for proper diagnosis.")
        
        # Downloadable Report
        report = f"Diagnosis: {'Benign' if prediction == 0 else 'Malignant'}\nConfidence: {confidence_score:.2f}%"
        st.download_button("📥 Download Report", report, file_name="cancer_prediction.txt")

if __name__ == "__main__":
    main()
