import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from sklearn.ensemble import RandomForestClassifier
from sklearn.datasets import load_iris, load_wine
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from sklearn.preprocessing import StandardScaler
import seaborn as sns
import matplotlib.pyplot as plt
from io import StringIO
import pickle
import base64

# Configure page
st.set_page_config(
    page_title="ML Model Deployment Hub",
    page_icon="🤖",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS for better styling
st.markdown("""
<style>
    .main-header {
        font-size: 2.5rem;
        color: #1f77b4;
        text-align: center;
        margin-bottom: 2rem;
    }
    .metric-container {
        background-color: #f0f2f6;
        padding: 1rem;
        border-radius: 0.5rem;
        margin: 0.5rem 0;
    }
    .prediction-box {
        background: linear-gradient(90deg, #667eea 0%, #764ba2 100%);
        color: white;
        padding: 2rem;
        border-radius: 1rem;
        text-align: center;
        margin: 1rem 0;
    }
</style>
""", unsafe_allow_html=True)

@st.cache_data
def load_dataset(dataset_name):
    """Load and prepare dataset"""
    if dataset_name == "Iris":
        data = load_iris()
        df = pd.DataFrame(data.data, columns=data.feature_names)
        df['target'] = data.target
        df['target_name'] = df['target'].map({0: data.target_names[0], 
                                            1: data.target_names[1], 
                                            2: data.target_names[2]})
        return df, data.feature_names, data.target_names
    elif dataset_name == "Wine":
        data = load_wine()
        df = pd.DataFrame(data.data, columns=data.feature_names)
        df['target'] = data.target
        df['target_name'] = df['target'].map({0: data.target_names[0], 
                                            1: data.target_names[1], 
                                            2: data.target_names[2]})
        return df, data.feature_names, data.target_names

@st.cache_resource
def train_model(df, feature_names):
    """Train the machine learning model"""
    X = df[feature_names]
    y = df['target']
    
    # Split data
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    
    # Scale features
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    
    # Train model
    model = RandomForestClassifier(n_estimators=100, random_state=42)
    model.fit(X_train_scaled, y_train)
    
    # Predictions
    y_pred = model.predict(X_test_scaled)
    
    return model, scaler, X_test, y_test, y_pred, X_train_scaled, y_train

def create_feature_importance_plot(model, feature_names):
    """Create feature importance visualization"""
    importance = model.feature_importances_
    
    fig = px.bar(
        x=importance,
        y=feature_names,
        orientation='h',
        title="Feature Importance",
        labels={'x': 'Importance Score', 'y': 'Features'}
    )
    fig.update_layout(height=400)
    return fig

def create_prediction_probability_plot(probabilities, target_names, prediction):
    """Create prediction probability visualization"""
    fig = go.Figure(data=[
        go.Bar(
            x=target_names,
            y=probabilities,
            marker_color=['#ff6b6b' if i != prediction else '#51cf66' for i in range(len(probabilities))],
            text=[f'{prob:.3f}' for prob in probabilities],
            textposition='auto'
        )
    ])
    
    fig.update_layout(
        title="Prediction Probabilities",
        xaxis_title="Classes",
        yaxis_title="Probability",
        height=400
    )
    return fig

def create_confusion_matrix_plot(y_test, y_pred, target_names):
    """Create confusion matrix heatmap"""
    cm = confusion_matrix(y_test, y_pred)
    
    fig = px.imshow(
        cm,
        x=target_names,
        y=target_names,
        color_continuous_scale='Blues',
        text_auto=True,
        aspect="auto",
        title="Confusion Matrix"
    )
    fig.update_layout(
        xaxis_title="Predicted",
        yaxis_title="Actual"
    )
    return fig

def main():
    # Header
    st.markdown('<h1 class="main-header">🤖 ML Model Deployment Hub</h1>', unsafe_allow_html=True)
    st.markdown("**Deploy, test, and understand your machine learning models interactively**")
    
    # Sidebar
    st.sidebar.header("🔧 Configuration")
    
    # Dataset selection
    dataset_choice = st.sidebar.selectbox(
        "Choose Dataset:",
        ["Iris", "Wine"]
    )
    
    # Load data
    df, feature_names, target_names = load_dataset(dataset_choice)
    
    # Train model
    with st.spinner("Training model..."):
        model, scaler, X_test, y_test, y_pred, X_train_scaled, y_train = train_model(df, feature_names)
    
    # Main tabs
    tab1, tab2, tab3, tab4 = st.tabs(["🎯 Make Predictions", "📊 Model Performance", "🔍 Model Insights", "📋 Dataset Explorer"])
    
    with tab1:
        st.header("Make Predictions")
        col1, col2 = st.columns([1, 1])
        
        with col1:
            st.subheader("Input Features")
            
            # Create input widgets for each feature
            user_inputs = {}
            for feature in feature_names:
                min_val = float(df[feature].min())
                max_val = float(df[feature].max())
                mean_val = float(df[feature].mean())
                
                user_inputs[feature] = st.slider(
                    f"{feature}:",
                    min_value=min_val,
                    max_value=max_val,
                    value=mean_val,
                    step=(max_val - min_val) / 100
                )
            
            # Prediction button
            if st.button("🎯 Make Prediction", type="primary"):
                # Prepare input data
                input_data = np.array([list(user_inputs.values())])
                input_scaled = scaler.transform(input_data)
                
                # Make prediction
                prediction = model.predict(input_scaled)[0]
                probabilities = model.predict_proba(input_scaled)[0]
                
                # Store in session state
                st.session_state.prediction = prediction
                st.session_state.probabilities = probabilities
                st.session_state.predicted_class = target_names[prediction]
                st.session_state.confidence = probabilities[prediction]
        
        with col2:
            st.subheader("Prediction Results")
            
            if hasattr(st.session_state, 'prediction'):
                # Display prediction
                st.markdown(f"""
                <div class="prediction-box">
                    <h2>Predicted Class: {st.session_state.predicted_class}</h2>
                    <h3>Confidence: {st.session_state.confidence:.3f}</h3>
                </div>
                """, unsafe_allow_html=True)
                
                # Probability plot
                prob_fig = create_prediction_probability_plot(
                    st.session_state.probabilities, 
                    target_names, 
                    st.session_state.prediction
                )
                st.plotly_chart(prob_fig, use_container_width=True)
                
                # Feature importance for this prediction
                st.subheader("Why this prediction?")
                input_df = pd.DataFrame([user_inputs])
                feature_contributions = model.feature_importances_ * np.abs(scaler.transform(input_df)[0])
                
                contrib_fig = px.bar(
                    x=feature_names,
                    y=feature_contributions,
                    title="Feature Contributions to Prediction"
                )
                st.plotly_chart(contrib_fig, use_container_width=True)
    
    with tab2:
        st.header("Model Performance")
        
        # Metrics
        accuracy = accuracy_score(y_test, y_pred)
        
        col1, col2, col3 = st.columns(3)
        with col1:
            st.metric("Accuracy", f"{accuracy:.3f}")
        with col2:
            st.metric("Test Samples", len(y_test))
        with col3:
            st.metric("Features", len(feature_names))
        
        # Confusion Matrix
        col1, col2 = st.columns(2)
        
        with col1:
            cm_fig = create_confusion_matrix_plot(y_test, y_pred, target_names)
            st.plotly_chart(cm_fig, use_container_width=True)
        
        with col2:
            # Classification report
            st.subheader("Classification Report")
            report = classification_report(y_test, y_pred, target_names=target_names, output_dict=True)
            report_df = pd.DataFrame(report).transpose()
            st.dataframe(report_df.round(3))
        
        # Model details
        st.subheader("Model Configuration")
        st.json({
            "Model Type": "Random Forest Classifier",
            "Number of Trees": model.n_estimators,
            "Random State": model.random_state,
            "Features Used": len(feature_names),
            "Classes": list(target_names)
        })
    
    with tab3:
        st.header("Model Insights")
        
        col1, col2 = st.columns(2)
        
        with col1:
            # Feature importance
            importance_fig = create_feature_importance_plot(model, feature_names)
            st.plotly_chart(importance_fig, use_container_width=True)
        
        with col2:
            # Feature correlation heatmap
            corr_matrix = df[feature_names].corr()
            fig = px.imshow(
                corr_matrix,
                text_auto=True,
                aspect="auto",
                title="Feature Correlation Matrix",
                color_continuous_scale="RdBu"
            )
            st.plotly_chart(fig, use_container_width=True)
        
        # Feature distributions by class
        st.subheader("Feature Distributions by Class")
        selected_feature = st.selectbox("Select feature to analyze:", feature_names)
        
        fig = px.violin(
            df, 
            x='target_name', 
            y=selected_feature,
            title=f"Distribution of {selected_feature} by Class",
            box=True
        )
        st.plotly_chart(fig, use_container_width=True)
        
        # Model explanation
        st.subheader("How the Model Works")
        st.info("""
        **Random Forest Classifier** uses multiple decision trees to make predictions:
        
        1. **Ensemble Learning**: Combines predictions from 100 decision trees
        2. **Feature Importance**: Each feature's contribution is calculated based on how much it decreases impurity
        3. **Voting**: Final prediction is based on majority vote from all trees
        4. **Robustness**: Less prone to overfitting compared to single decision trees
        
        The confidence score represents the proportion of trees that voted for the predicted class.
        """)
    
    with tab4:
        st.header("Dataset Explorer")
        
        col1, col2 = st.columns([2, 1])
        
        with col1:
            st.subheader("Dataset Overview")
            st.dataframe(df.head(10))
            
            # Dataset statistics
            st.subheader("Statistical Summary")
            st.dataframe(df[feature_names].describe())
        
        with col2:
            st.subheader("Dataset Info")
            st.info(f"""
            **Dataset**: {dataset_choice}
            
            **Samples**: {len(df)}
            
            **Features**: {len(feature_names)}
            
            **Classes**: {len(target_names)}
            
            **Target Distribution**:
            """)
            
            target_counts = df['target_name'].value_counts()
            fig = px.pie(
                values=target_counts.values,
                names=target_counts.index,
                title="Class Distribution"
            )
            st.plotly_chart(fig, use_container_width=True)
        
        # Download options
        st.subheader("Export Options")
        col1, col2, col3 = st.columns(3)
        
        with col1:
            if st.button("📥 Download Dataset"):
                csv = df.to_csv(index=False)
                st.download_button(
                    label="Download CSV",
                    data=csv,
                    file_name=f"{dataset_choice.lower()}_dataset.csv",
                    mime="text/csv"
                )
        
        with col2:
            if st.button("📥 Download Model"):
                # Serialize model
                model_bytes = pickle.dumps({"model": model, "scaler": scaler})
                st.download_button(
                    label="Download Model",
                    data=model_bytes,
                    file_name=f"{dataset_choice.lower()}_model.pkl",
                    mime="application/octet-stream"
                )
        
        with col3:
            if st.button("📊 Download Predictions"):
                predictions_df = pd.DataFrame({
                    'Actual': y_test,
                    'Predicted': y_pred,
                    'Correct': y_test == y_pred
                })
                csv = predictions_df.to_csv(index=False)
                st.download_button(
                    label="Download Predictions",
                    data=csv,
                    file_name=f"{dataset_choice.lower()}_predictions.csv",
                    mime="text/csv"
                )

if __name__ == "__main__":
    main()