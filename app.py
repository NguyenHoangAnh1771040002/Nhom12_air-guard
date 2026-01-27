"""
AIR GUARD Dashboard - Dự báo PM2.5 và Cảnh báo AQI
Streamlit Dashboard for Semi-Supervised AQI Classification
"""

import streamlit as st
import pandas as pd
import numpy as np
import json
from pathlib import Path
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots

# Page config
st.set_page_config(
    page_title="AIR GUARD - Dự báo AQI",
    page_icon="🌬️",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Paths
PROJECT_ROOT = Path(__file__).parent
DATA_DIR = PROJECT_ROOT / "data" / "processed"

# Load data functions
@st.cache_data
def load_metrics():
    """Load all metrics files"""
    metrics = {}
    
    # Baseline
    baseline_path = DATA_DIR / "metrics.json"
    if baseline_path.exists():
        with open(baseline_path, "r") as f:
            metrics["baseline"] = json.load(f)
    
    # Self-Training
    st_path = DATA_DIR / "metrics_self_training.json"
    if st_path.exists():
        with open(st_path, "r") as f:
            metrics["self_training"] = json.load(f)
    
    # Co-Training
    ct_path = DATA_DIR / "metrics_co_training.json"
    if ct_path.exists():
        with open(ct_path, "r") as f:
            metrics["co_training"] = json.load(f)
    
    return metrics

@st.cache_data
def load_predictions():
    """Load prediction samples"""
    preds = {}
    
    for name, filename in [
        ("baseline", "predictions_sample.csv"),
        ("self_training", "predictions_self_training_sample.csv"),
        ("co_training", "predictions_co_training_sample.csv")
    ]:
        path = DATA_DIR / filename
        if path.exists():
            preds[name] = pd.read_csv(path)
    
    return preds

@st.cache_data
def load_alerts():
    """Load alert samples"""
    alerts = {}
    
    for name, filename in [
        ("self_training", "alerts_self_training_sample.csv"),
        ("co_training", "alerts_co_training_sample.csv")
    ]:
        path = DATA_DIR / filename
        if path.exists():
            alerts[name] = pd.read_csv(path)
    
    return alerts

# Sidebar
st.sidebar.title("🌬️ AIR GUARD")
st.sidebar.markdown("**Dự báo PM2.5 & Cảnh báo AQI**")
st.sidebar.markdown("---")

page = st.sidebar.radio(
    "📌 Chọn trang",
    ["🏠 Tổng quan", "📊 So sánh mô hình", "📈 Diễn biến huấn luyện", "🚨 Cảnh báo AQI", "📋 Chi tiết dự đoán"]
)

# Load data
metrics = load_metrics()
predictions = load_predictions()
alerts = load_alerts()

# AQI Class colors
AQI_COLORS = {
    "Good": "#00e400",
    "Moderate": "#ffff00",
    "Unhealthy_for_Sensitive_Groups": "#ff7e00",
    "Unhealthy": "#ff0000",
    "Very_Unhealthy": "#8f3f97",
    "Hazardous": "#7e0023"
}

# ==================== PAGE: Tổng quan ====================
if page == "🏠 Tổng quan":
    st.title("🌬️ AIR GUARD - Hệ thống Dự báo Chất lượng Không khí")
    
    st.markdown("""
    ### 📋 Giới thiệu dự án
    
    Hệ thống **AIR GUARD** sử dụng các phương pháp **học bán giám sát** (Semi-Supervised Learning) 
    để dự báo chất lượng không khí (AQI) dựa trên nồng độ PM2.5, ngay cả khi thiếu dữ liệu có nhãn.
    
    #### 🎯 Các phương pháp đã triển khai:
    - **Baseline**: Mô hình có giám sát truyền thống (100% dữ liệu có nhãn)
    - **Self-Training**: Tự huấn luyện với pseudo-labels (~8.7% dữ liệu có nhãn ban đầu)
    - **Co-Training**: Đồng huấn luyện với 2 views đặc trưng khác nhau
    """)
    
    # Summary metrics
    st.markdown("### 📊 Tổng quan kết quả")
    
    col1, col2, col3 = st.columns(3)
    
    if "baseline" in metrics:
        with col1:
            st.metric(
                label="🎯 Baseline Accuracy",
                value=f"{metrics['baseline']['accuracy']:.2%}"
            )
            st.metric(
                label="📈 Baseline F1-macro",
                value=f"{metrics['baseline']['f1_macro']:.2%}"
            )
    
    if "self_training" in metrics:
        with col2:
            st.metric(
                label="🔄 Self-Training Accuracy",
                value=f"{metrics['self_training']['test_metrics']['accuracy']:.2%}",
                delta=f"{metrics['self_training']['test_metrics']['accuracy'] - metrics['baseline']['accuracy']:.2%}" if "baseline" in metrics else None
            )
            st.metric(
                label="📈 Self-Training F1-macro",
                value=f"{metrics['self_training']['test_metrics']['f1_macro']:.2%}",
                delta=f"{metrics['self_training']['test_metrics']['f1_macro'] - metrics['baseline']['f1_macro']:.2%}" if "baseline" in metrics else None
            )
    
    if "co_training" in metrics:
        with col3:
            st.metric(
                label="🤝 Co-Training Accuracy",
                value=f"{metrics['co_training']['test_metrics']['accuracy']:.2%}",
                delta=f"{metrics['co_training']['test_metrics']['accuracy'] - metrics['baseline']['accuracy']:.2%}" if "baseline" in metrics else None
            )
            st.metric(
                label="📈 Co-Training F1-macro",
                value=f"{metrics['co_training']['test_metrics']['f1_macro']:.2%}",
                delta=f"{metrics['co_training']['test_metrics']['f1_macro'] - metrics['baseline']['f1_macro']:.2%}" if "baseline" in metrics else None
            )
    
    # AQI Classes explanation
    st.markdown("### 🏷️ Các mức AQI")
    
    aqi_df = pd.DataFrame({
        "Mức AQI": ["Good", "Moderate", "Unhealthy for Sensitive Groups", "Unhealthy", "Very Unhealthy", "Hazardous"],
        "Tiếng Việt": ["Tốt", "Trung bình", "Không tốt cho nhóm nhạy cảm", "Không lành mạnh", "Rất không lành mạnh", "Nguy hại"],
        "PM2.5 (µg/m³)": ["0-9", "9.1-35.4", "35.5-55.4", "55.5-125.4", "125.5-225.4", "225.5+"],
        "Màu": ["🟢", "🟡", "🟠", "🔴", "🟣", "🟤"]
    })
    st.table(aqi_df)

# ==================== PAGE: So sánh mô hình ====================
elif page == "📊 So sánh mô hình":
    st.title("📊 So sánh hiệu năng các mô hình")
    
    # Comparison table
    comparison_data = []
    
    if "baseline" in metrics:
        comparison_data.append({
            "Phương pháp": "Baseline (100% labels)",
            "Test Accuracy": metrics["baseline"]["accuracy"],
            "Test F1-macro": metrics["baseline"]["f1_macro"]
        })
    
    if "self_training" in metrics:
        comparison_data.append({
            "Phương pháp": f"Self-Training (τ={metrics['self_training']['st_cfg']['tau']})",
            "Test Accuracy": metrics["self_training"]["test_metrics"]["accuracy"],
            "Test F1-macro": metrics["self_training"]["test_metrics"]["f1_macro"]
        })
    
    if "co_training" in metrics:
        comparison_data.append({
            "Phương pháp": f"Co-Training (τ={metrics['co_training']['ct_cfg']['tau']})",
            "Test Accuracy": metrics["co_training"]["test_metrics"]["accuracy"],
            "Test F1-macro": metrics["co_training"]["test_metrics"]["f1_macro"]
        })
    
    if comparison_data:
        df_compare = pd.DataFrame(comparison_data)
        
        # Bar charts
        col1, col2 = st.columns(2)
        
        with col1:
            fig_acc = px.bar(
                df_compare, 
                x="Phương pháp", 
                y="Test Accuracy",
                color="Phương pháp",
                title="So sánh Test Accuracy",
                text_auto='.2%'
            )
            fig_acc.update_layout(showlegend=False)
            st.plotly_chart(fig_acc, use_container_width=True)
        
        with col2:
            fig_f1 = px.bar(
                df_compare, 
                x="Phương pháp", 
                y="Test F1-macro",
                color="Phương pháp",
                title="So sánh Test F1-macro",
                text_auto='.2%'
            )
            fig_f1.update_layout(showlegend=False)
            st.plotly_chart(fig_f1, use_container_width=True)
        
        # Table
        st.markdown("### 📋 Bảng so sánh chi tiết")
        df_display = df_compare.copy()
        df_display["Test Accuracy"] = df_display["Test Accuracy"].apply(lambda x: f"{x:.2%}")
        df_display["Test F1-macro"] = df_display["Test F1-macro"].apply(lambda x: f"{x:.2%}")
        st.dataframe(df_display, use_container_width=True)
        
        # Insights
        st.markdown("### 💡 Nhận xét")
        
        best_f1_method = df_compare.loc[df_compare["Test F1-macro"].idxmax(), "Phương pháp"]
        best_f1_value = df_compare["Test F1-macro"].max()
        
        st.success(f"""
        **Phương pháp tốt nhất (theo F1-macro):** {best_f1_method} với F1-macro = {best_f1_value:.2%}
        
        **Kết luận:**
        - Self-Training với ~8.7% dữ liệu có nhãn ban đầu đạt F1-macro cao hơn baseline
        - Việc sử dụng dữ liệu không nhãn giúp cải thiện chất lượng phân loại
        - Co-Training cần thiết kế views cẩn thận để đạt hiệu quả tốt
        """)

# ==================== PAGE: Diễn biến huấn luyện ====================
elif page == "📈 Diễn biến huấn luyện":
    st.title("📈 Diễn biến quá trình huấn luyện")
    
    tab1, tab2 = st.tabs(["🔄 Self-Training", "🤝 Co-Training"])
    
    with tab1:
        if "self_training" in metrics and "history" in metrics["self_training"]:
            history = pd.DataFrame(metrics["self_training"]["history"])
            
            col1, col2 = st.columns(2)
            
            with col1:
                fig1 = px.line(
                    history, 
                    x="iter", 
                    y="val_f1_macro",
                    markers=True,
                    title="Validation F1-macro qua các vòng"
                )
                fig1.update_layout(xaxis_title="Vòng lặp", yaxis_title="Val F1-macro")
                st.plotly_chart(fig1, use_container_width=True)
            
            with col2:
                fig2 = px.bar(
                    history, 
                    x="iter", 
                    y="new_pseudo",
                    title="Số pseudo-labels mới mỗi vòng"
                )
                fig2.update_layout(xaxis_title="Vòng lặp", yaxis_title="Số mẫu mới")
                st.plotly_chart(fig2, use_container_width=True)
            
            st.dataframe(history, use_container_width=True)
        else:
            st.warning("Chưa có dữ liệu Self-Training")
    
    with tab2:
        if "co_training" in metrics and "history" in metrics["co_training"]:
            history = pd.DataFrame(metrics["co_training"]["history"])
            
            col1, col2 = st.columns(2)
            
            with col1:
                fig1 = px.line(
                    history, 
                    x="iter", 
                    y="val_f1_macro",
                    markers=True,
                    title="Validation F1-macro qua các vòng"
                )
                fig1.update_layout(xaxis_title="Vòng lặp", yaxis_title="Val F1-macro")
                st.plotly_chart(fig1, use_container_width=True)
            
            with col2:
                fig2 = px.bar(
                    history, 
                    x="iter", 
                    y="new_pseudo",
                    title="Số pseudo-labels mới mỗi vòng"
                )
                fig2.update_layout(xaxis_title="Vòng lặp", yaxis_title="Số mẫu mới")
                st.plotly_chart(fig2, use_container_width=True)
            
            st.dataframe(history, use_container_width=True)
        else:
            st.warning("Chưa có dữ liệu Co-Training")

# ==================== PAGE: Cảnh báo AQI ====================
elif page == "🚨 Cảnh báo AQI":
    st.title("🚨 Cảnh báo Chất lượng Không khí")
    
    method = st.selectbox(
        "Chọn phương pháp:",
        ["self_training", "co_training"],
        format_func=lambda x: "Self-Training" if x == "self_training" else "Co-Training"
    )
    
    if method in alerts and alerts[method] is not None:
        alert_df = alerts[method]
        
        if "is_alert" in alert_df.columns:
            # Alert statistics
            total_samples = len(alert_df)
            alert_samples = alert_df["is_alert"].sum()
            alert_rate = alert_samples / total_samples
            
            col1, col2, col3 = st.columns(3)
            with col1:
                st.metric("Tổng số mẫu", f"{total_samples:,}")
            with col2:
                st.metric("Số mẫu cảnh báo", f"{alert_samples:,}", delta=f"{alert_rate:.1%}")
            with col3:
                st.metric("Tỷ lệ cảnh báo", f"{alert_rate:.1%}")
            
            # Alert by station
            if "station" in alert_df.columns:
                st.markdown("### 📍 Cảnh báo theo trạm")
                
                station_alerts = alert_df.groupby("station")["is_alert"].agg(["sum", "count"])
                station_alerts["rate"] = station_alerts["sum"] / station_alerts["count"]
                station_alerts = station_alerts.reset_index()
                station_alerts.columns = ["Trạm", "Số cảnh báo", "Tổng mẫu", "Tỷ lệ"]
                
                fig = px.bar(
                    station_alerts.sort_values("Tỷ lệ", ascending=False),
                    x="Trạm",
                    y="Tỷ lệ",
                    title="Tỷ lệ cảnh báo theo trạm",
                    text_auto='.1%'
                )
                st.plotly_chart(fig, use_container_width=True)
                
                st.dataframe(station_alerts, use_container_width=True)
            
            # Show alert samples
            st.markdown("### 📋 Mẫu cảnh báo")
            alert_only = alert_df[alert_df["is_alert"] == True].head(100)
            st.dataframe(alert_only, use_container_width=True)
        else:
            st.warning("Không có cột 'is_alert' trong dữ liệu")
    else:
        st.warning(f"Chưa có dữ liệu cảnh báo cho {method}")

# ==================== PAGE: Chi tiết dự đoán ====================
elif page == "📋 Chi tiết dự đoán":
    st.title("📋 Chi tiết dự đoán")
    
    method = st.selectbox(
        "Chọn phương pháp:",
        list(predictions.keys()),
        format_func=lambda x: {"baseline": "Baseline", "self_training": "Self-Training", "co_training": "Co-Training"}.get(x, x)
    )
    
    if method in predictions and predictions[method] is not None:
        pred_df = predictions[method]
        
        st.markdown(f"### 📊 Dự đoán - {method.replace('_', ' ').title()}")
        
        # Class distribution
        if "y_pred" in pred_df.columns:
            col1, col2 = st.columns(2)
            
            with col1:
                pred_counts = pred_df["y_pred"].value_counts()
                fig = px.pie(
                    values=pred_counts.values,
                    names=pred_counts.index,
                    title="Phân bố dự đoán AQI",
                    color_discrete_sequence=px.colors.qualitative.Set2
                )
                st.plotly_chart(fig, use_container_width=True)
            
            with col2:
                if "y_true" in pred_df.columns:
                    true_counts = pred_df["y_true"].value_counts()
                    fig = px.pie(
                        values=true_counts.values,
                        names=true_counts.index,
                        title="Phân bố thực tế AQI",
                        color_discrete_sequence=px.colors.qualitative.Set2
                    )
                    st.plotly_chart(fig, use_container_width=True)
        
        # Filter options
        st.markdown("### 🔍 Lọc dữ liệu")
        
        col1, col2 = st.columns(2)
        
        with col1:
            if "station" in pred_df.columns:
                stations = ["Tất cả"] + list(pred_df["station"].unique())
                selected_station = st.selectbox("Chọn trạm:", stations)
        
        with col2:
            if "y_pred" in pred_df.columns:
                classes = ["Tất cả"] + list(pred_df["y_pred"].unique())
                selected_class = st.selectbox("Chọn mức AQI:", classes)
        
        # Filter data
        filtered_df = pred_df.copy()
        if "station" in pred_df.columns and selected_station != "Tất cả":
            filtered_df = filtered_df[filtered_df["station"] == selected_station]
        if "y_pred" in pred_df.columns and selected_class != "Tất cả":
            filtered_df = filtered_df[filtered_df["y_pred"] == selected_class]
        
        st.markdown(f"**Hiển thị {len(filtered_df):,} mẫu**")
        st.dataframe(filtered_df.head(500), use_container_width=True)
    else:
        st.warning(f"Chưa có dữ liệu dự đoán cho {method}")

# Footer
st.sidebar.markdown("---")
st.sidebar.markdown("""
**📚 AIR GUARD Project**

Mini Project: Semi-Supervised AQI Classification

© 2026 - Data Mining
""")
