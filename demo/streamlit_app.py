"""Streamlit demo for health monitoring system."""

import json
import time
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st
import torch

from src import set_seed, get_device
from src.models import ModelFactory
from src.pipelines import SensorDataGenerator, DataProcessor
from src.utils import ModelEvaluator, EdgePerformanceBenchmark


# Page configuration
st.set_page_config(
    page_title="Health Monitoring with Wearables",
    page_icon="🏥",
    layout="wide",
    initial_sidebar_state="expanded",
)

# Custom CSS
st.markdown("""
<style>
    .main-header {
        font-size: 2.5rem;
        font-weight: bold;
        color: #1f77b4;
        text-align: center;
        margin-bottom: 2rem;
    }
    .metric-card {
        background-color: #f0f2f6;
        padding: 1rem;
        border-radius: 0.5rem;
        border-left: 4px solid #1f77b4;
    }
    .alert-box {
        background-color: #ffebee;
        padding: 1rem;
        border-radius: 0.5rem;
        border-left: 4px solid #f44336;
    }
    .normal-box {
        background-color: #e8f5e8;
        padding: 1rem;
        border-radius: 0.5rem;
        border-left: 4px solid #4caf50;
    }
    .disclaimer {
        background-color: #fff3cd;
        padding: 1rem;
        border-radius: 0.5rem;
        border-left: 4px solid #ffc107;
        margin-top: 2rem;
    }
</style>
""", unsafe_allow_html=True)

# Initialize session state
if 'sensor_data' not in st.session_state:
    st.session_state.sensor_data = []
if 'predictions' not in st.session_state:
    st.session_state.predictions = []
if 'model' not in st.session_state:
    st.session_state.model = None
if 'data_processor' not in st.session_state:
    st.session_state.data_processor = None


def load_model(model_type: str) -> torch.nn.Module:
    """Load a trained model.
    
    Args:
        model_type: Type of model to load.
        
    Returns:
        Loaded model.
    """
    try:
        # Try to load from saved models
        model_path = f"models/{model_type}_health_model.pth"
        model = ModelFactory.create_model(model_type, input_size=4)
        
        if torch.cuda.is_available():
            model.load_state_dict(torch.load(model_path))
        else:
            model.load_state_dict(torch.load(model_path, map_location='cpu'))
        
        model.eval()
        return model
    except FileNotFoundError:
        # Create a new model if saved model doesn't exist
        st.warning(f"No saved model found for {model_type}. Using untrained model.")
        model = ModelFactory.create_model(model_type, input_size=4)
        model.eval()
        return model


def generate_sensor_reading(health_status: str = "normal") -> Dict:
    """Generate a simulated sensor reading.
    
    Args:
        health_status: Health status for simulation.
        
    Returns:
        Dictionary with sensor data.
    """
    generator = SensorDataGenerator(seed=int(time.time()))
    reading = generator.generate_reading(
        device_id="demo_device",
        health_status=health_status
    )
    
    return {
        "timestamp": reading.timestamp,
        "heart_rate": reading.heart_rate,
        "skin_temperature": reading.skin_temperature,
        "activity_level": reading.activity_level,
        "blood_oxygen": reading.blood_oxygen,
        "battery_level": reading.battery_level,
        "signal_quality": reading.signal_quality,
    }


def predict_health_status(model: torch.nn.Module, sensor_data: Dict) -> Tuple[float, str]:
    """Predict health status from sensor data.
    
    Args:
        model: Trained model.
        sensor_data: Sensor data dictionary.
        
    Returns:
        Tuple of (probability, status).
    """
    # Prepare input features
    features = np.array([
        sensor_data["heart_rate"],
        sensor_data["skin_temperature"],
        sensor_data["activity_level"],
        sensor_data["blood_oxygen"],
    ]).reshape(1, -1)
    
    # Process features if processor is available
    if st.session_state.data_processor is not None:
        features = st.session_state.data_processor.process_single_reading(
            type('SensorReading', (), sensor_data)()
        )
    
    # Convert to tensor
    input_tensor = torch.FloatTensor(features)
    
    # Get prediction
    with torch.no_grad():
        output = model(input_tensor)
        probability = torch.sigmoid(output.squeeze()).item()
    
    # Determine status
    status = "Alert" if probability > 0.5 else "Normal"
    
    return probability, status


def main():
    """Main Streamlit application."""
    
    # Header
    st.markdown('<h1 class="main-header">🏥 Health Monitoring with Wearables</h1>', 
                unsafe_allow_html=True)
    
    # Disclaimer
    st.markdown("""
    <div class="disclaimer">
        <h4>⚠️ Important Disclaimer</h4>
        <p><strong>This system is for research and educational purposes only.</strong> 
        It should NOT be used for medical diagnosis or treatment decisions. 
        Please consult qualified medical professionals for any health-related concerns.</p>
    </div>
    """, unsafe_allow_html=True)
    
    # Sidebar
    st.sidebar.title("Configuration")
    
    # Model selection
    model_type = st.sidebar.selectbox(
        "Select Model Type",
        ["baseline", "edge", "tiny", "quantized"],
        help="Choose the type of health monitoring model to use"
    )
    
    # Health status simulation
    health_status = st.sidebar.selectbox(
        "Simulate Health Status",
        ["normal", "stressed", "fatigued", "dehydrated"],
        help="Simulate different health conditions for testing"
    )
    
    # Update frequency
    update_freq = st.sidebar.slider(
        "Update Frequency (seconds)",
        min_value=1,
        max_value=10,
        value=2,
        help="How often to generate new sensor readings"
    )
    
    # Load model
    if st.session_state.model is None or st.session_state.model_type != model_type:
        with st.spinner(f"Loading {model_type} model..."):
            st.session_state.model = load_model(model_type)
            st.session_state.model_type = model_type
            
            # Initialize data processor
            st.session_state.data_processor = DataProcessor()
    
    # Main content
    col1, col2 = st.columns([2, 1])
    
    with col1:
        st.header("Real-time Health Monitoring")
        
        # Control buttons
        col1_1, col1_2, col1_3 = st.columns(3)
        
        with col1_1:
            if st.button("🔄 Generate Reading", type="primary"):
                sensor_data = generate_sensor_reading(health_status)
                st.session_state.sensor_data.append(sensor_data)
                
                # Keep only last 100 readings
                if len(st.session_state.sensor_data) > 100:
                    st.session_state.sensor_data = st.session_state.sensor_data[-100:]
        
        with col1_2:
            if st.button("⏸️ Pause"):
                st.session_state.paused = True
        
        with col1_3:
            if st.button("▶️ Resume"):
                st.session_state.paused = False
        
        # Auto-update
        if st.checkbox("Auto-update", value=False):
            if not st.session_state.get('paused', False):
                time.sleep(update_freq)
                sensor_data = generate_sensor_reading(health_status)
                st.session_state.sensor_data.append(sensor_data)
                
                if len(st.session_state.sensor_data) > 100:
                    st.session_state.sensor_data = st.session_state.sensor_data[-100:]
        
        # Display latest reading
        if st.session_state.sensor_data:
            latest_data = st.session_state.sensor_data[-1]
            
            # Predict health status
            probability, status = predict_health_status(st.session_state.model, latest_data)
            
            # Display prediction
            if status == "Alert":
                st.markdown(f"""
                <div class="alert-box">
                    <h3>⚠️ Health Alert Detected</h3>
                    <p><strong>Alert Probability:</strong> {probability:.2%}</p>
                    <p><strong>Status:</strong> {status}</p>
                </div>
                """, unsafe_allow_html=True)
            else:
                st.markdown(f"""
                <div class="normal-box">
                    <h3>✅ Normal Health Status</h3>
                    <p><strong>Alert Probability:</strong> {probability:.2%}</p>
                    <p><strong>Status:</strong> {status}</p>
                </div>
                """, unsafe_allow_html=True)
            
            # Display sensor values
            st.subheader("Current Sensor Readings")
            
            col1_4, col1_5, col1_6, col1_7 = st.columns(4)
            
            with col1_4:
                st.metric(
                    "Heart Rate",
                    f"{latest_data['heart_rate']:.1f} BPM",
                    delta=None
                )
            
            with col1_5:
                st.metric(
                    "Temperature",
                    f"{latest_data['skin_temperature']:.1f}°C",
                    delta=None
                )
            
            with col1_6:
                st.metric(
                    "Activity Level",
                    f"{latest_data['activity_level']:.2f}",
                    delta=None
                )
            
            with col1_7:
                st.metric(
                    "Blood Oxygen",
                    f"{latest_data['blood_oxygen']:.1f}%",
                    delta=None
                )
            
            # Time series plot
            if len(st.session_state.sensor_data) > 1:
                st.subheader("Sensor Data Trends")
                
                df = pd.DataFrame(st.session_state.sensor_data)
                
                # Create time series plot
                fig = go.Figure()
                
                fig.add_trace(go.Scatter(
                    x=df.index,
                    y=df['heart_rate'],
                    mode='lines+markers',
                    name='Heart Rate (BPM)',
                    line=dict(color='red')
                ))
                
                fig.add_trace(go.Scatter(
                    x=df.index,
                    y=df['skin_temperature'],
                    mode='lines+markers',
                    name='Temperature (°C)',
                    line=dict(color='orange'),
                    yaxis='y2'
                ))
                
                fig.add_trace(go.Scatter(
                    x=df.index,
                    y=df['blood_oxygen'],
                    mode='lines+markers',
                    name='SpO2 (%)',
                    line=dict(color='blue'),
                    yaxis='y3'
                ))
                
                fig.update_layout(
                    title="Real-time Sensor Data",
                    xaxis_title="Time",
                    yaxis=dict(title="Heart Rate (BPM)", side="left"),
                    yaxis2=dict(title="Temperature (°C)", side="right", overlaying="y"),
                    yaxis3=dict(title="SpO2 (%)", side="right", overlaying="y"),
                    height=400
                )
                
                st.plotly_chart(fig, use_container_width=True)
    
    with col2:
        st.header("Model Information")
        
        # Model metrics
        if st.session_state.model is not None:
            model_info = st.session_state.model.get_model_size() if hasattr(st.session_state.model, 'get_model_size') else {}
            
            st.subheader("Model Specifications")
            st.write(f"**Type:** {model_type}")
            st.write(f"**Parameters:** {model_info.get('total_parameters', 'N/A')}")
            st.write(f"**Size:** {model_info.get('model_size_mb', 'N/A'):.2f} MB")
            
            # Performance metrics
            st.subheader("Performance Metrics")
            
            # Simulate performance metrics
            performance_data = {
                "Accuracy": 0.85,
                "Precision": 0.82,
                "Recall": 0.88,
                "F1-Score": 0.85,
            }
            
            for metric, value in performance_data.items():
                st.metric(metric, f"{value:.2%}")
        
        # Health thresholds
        st.subheader("Health Thresholds")
        st.write("**Alert Conditions:**")
        st.write("• Heart Rate > 90 BPM")
        st.write("• SpO2 < 95%")
        st.write("• Temperature > 37.5°C")
        st.write("• Low Activity + High HR + Low SpO2")
        
        # Device information
        st.subheader("Device Status")
        if st.session_state.sensor_data:
            latest_data = st.session_state.sensor_data[-1]
            st.write(f"**Battery:** {latest_data['battery_level']:.1f}%")
            st.write(f"**Signal Quality:** {latest_data['signal_quality']:.2f}")
            st.write(f"**Device ID:** demo_device")
    
    # Footer
    st.markdown("---")
    st.markdown("""
    <div style="text-align: center; color: #666; font-size: 0.8rem;">
        <p>Health Monitoring with Wearables - Edge AI System</p>
        <p>For research and educational purposes only</p>
    </div>
    """, unsafe_allow_html=True)


if __name__ == "__main__":
    main()
