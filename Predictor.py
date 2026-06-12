#!/usr/bin/env python
# coding: utf-8

# In[ ]:

import streamlit as st
import pandas as pd
import numpy as np
import joblib

st.set_page_config(
    page_title="Predictor of FRP-concrete bond strength",
    page_icon="🏗️",
    layout="wide"
)

# ==================== 字体大小设置 ====================
with st.sidebar:
    st.header("📌 Model Information")
    st.markdown("""
     - **Input**: 10 features (6 original + 4 derived)  
     - **Output**: Bond strength $P_u$ (kN)  
     - **Algorithm**: GradientBoosting Regressor
    """)
    st.divider()
    st.caption("Model file: Prediction of FRP-concrete bond strength.pkl")
    
    st.subheader("⚙️ Appearance")
    font_size = st.slider(
        "Font Size (px)", 
        min_value=12, 
        max_value=24, 
        value=16, 
        step=1,
        help="Adjust the font size of the main content (including labels, descriptions, and results)"
    )
    
    st.markdown(f"""
    <style>
    .main, .stApp, .stMarkdown, .stAlert, .stException {{
        font-size: {font_size}px !important;
    }}
    h1 {{
        font-size: {font_size + 8}px !important;
    }}
    h2, h3, .stMarkdown h2, .stMarkdown h3 {{
        font-size: {font_size + 4}px !important;
    }}
    p, li, .stMarkdown p, .stMarkdown li, .stMarkdown div, .stMarkdown span {{
        font-size: {font_size}px !important;
    }}
    .stNumberInput label, .stTextInput label, .stSelectbox label, .stSlider label {{
        font-size: {font_size}px !important;
        font-weight: 500 !important;
    }}
    .stButton button, .stDownloadButton button {{
        font-size: {font_size}px !important;
    }}
    .stMetric label, .stMetric value {{
        font-size: {font_size}px !important;
    }}
    .stAlert div, .stAlert p {{
        font-size: {font_size}px !important;
    }}
    .css-1d391kg, .sidebar .stMarkdown, .sidebar .stCaption {{
        font-size: {font_size - 1}px !important;
    }}
    .streamlit-expanderHeader, .streamlit-expanderContent {{
        font-size: {font_size}px !important;
    }}
    .dataframe, .stTable td, .stTable th {{
        font-size: {font_size - 1}px !important;
    }}
    </style>
    """, unsafe_allow_html=True)

# 检查 GradientBoostingRegressor 是否可用
try:
    from sklearn.ensemble import GradientBoostingRegressor
    st.sidebar.success("✅ GradientBoosting backend ready")
except ImportError:
    st.sidebar.error("❌ GradientBoostingRegressor not available. Please ensure scikit-learn is installed.")
    st.stop()

# 加载模型
@st.cache_resource
def load_model():
    return joblib.load('Prediction of FRP-concrete bond strength.pkl')

try:
    model = load_model()
    st.success("✅ Model loaded successfully!", icon="🎉")
except Exception as e:
    st.error(f"❌ Model loading failed: {e}\nEnsure the model file exists and GradientBoostingRegressor is installed.")
    st.stop()

st.title("🏗️ Predictor of FRP-concrete bond strength")
st.markdown("""
This application is based on the **GradientBoosting Regressor** model to predict the interface bond strength between FRP and concrete.
Please enter the **6 original parameters bc, fc', Ef, tf, bf, lf**; 
the remaining 4 derived features will be computed automatically : Kf = Ef·tf, bf/bc = bf/bc, Af = bf·lf, Df = bf·tf/lf .
""")

# ==================== 定义模型训练参数范围（10个参数） ====================
# 请根据您的实际训练数据调整这些范围
PARAM_RANGES = {
    'bc':   {'min': 75.0, 'max': 600.0, 'name': 'bc (Width of the concrete substrate / mm)'},
    'fc':   {'min': 8.0,  'max': 74.5,  'name': 'fc (Compressive strength of concrete / MPa)'},
    'Ef':   {'min': 22.5, 'max': 425.1,  'name': 'Ef (Elastic modulus of FRP sheet / GPa)'},
    'tf':   {'min': 0.083,   'max':6.0,   'name': 'tf (Thickness of FRP sheet / mm)'},
    'bf':   {'min': 10.0,  'max': 200.0,  'name': 'bf (Width of FRP sheet / mm)'},
    'lf':   {'min': 20.0, 'max': 1524.0, 'name': 'lf (Bond length of FRP sheet / mm)'},
    'Kf':   {'min': 8.52, 'max': 632.4,'name': 'Kf = Ef·tf (Stiffness of FRP sheet / GPa·mm)'},
    'bf_bc':{'min': 0.0333,   'max': 1.0,    'name': 'bf/bc (Width ratio between FRP sheet and concrete substrate)'},
    'Af':   {'min': 500.0,'max': 240000.0,'name': 'Af = bf·lf (Bond area of FRP sheet / mm²)'},
    'Df':   {'min': 0.00786,   'max': 1.4,  'name': 'Df = bf·tf / lf (Slenderness ratio of FRP sheet)'}
}

def check_parameters_out_of_range(bc, fc, Ef, tf, bf, lf, Kf, bf_bc, Af, Df):
    """检查所有10个参数是否超出模型训练范围"""
    out_of_range = []
    if bc < PARAM_RANGES['bc']['min'] or bc > PARAM_RANGES['bc']['max']:
        out_of_range.append(PARAM_RANGES['bc']['name'])
    if fc < PARAM_RANGES['fc']['min'] or fc > PARAM_RANGES['fc']['max']:
        out_of_range.append(PARAM_RANGES['fc']['name'])
    if Ef < PARAM_RANGES['Ef']['min'] or Ef > PARAM_RANGES['Ef']['max']:
        out_of_range.append(PARAM_RANGES['Ef']['name'])
    if tf < PARAM_RANGES['tf']['min'] or tf > PARAM_RANGES['tf']['max']:
        out_of_range.append(PARAM_RANGES['tf']['name'])
    if bf < PARAM_RANGES['bf']['min'] or bf > PARAM_RANGES['bf']['max']:
        out_of_range.append(PARAM_RANGES['bf']['name'])
    if lf < PARAM_RANGES['lf']['min'] or lf > PARAM_RANGES['lf']['max']:
        out_of_range.append(PARAM_RANGES['lf']['name'])
    if Kf < PARAM_RANGES['Kf']['min'] or Kf > PARAM_RANGES['Kf']['max']:
        out_of_range.append(PARAM_RANGES['Kf']['name'])
    if bf_bc < PARAM_RANGES['bf_bc']['min'] or bf_bc > PARAM_RANGES['bf_bc']['max']:
        out_of_range.append(PARAM_RANGES['bf_bc']['name'])
    if Af < PARAM_RANGES['Af']['min'] or Af > PARAM_RANGES['Af']['max']:
        out_of_range.append(PARAM_RANGES['Af']['name'])
    if Df < PARAM_RANGES['Df']['min'] or Df > PARAM_RANGES['Df']['max']:
        out_of_range.append(PARAM_RANGES['Df']['name'])
    return out_of_range

st.header("📊 Input features")
col1, col2 = st.columns(2)

with col1:
    bc = st.number_input("**bc** - Width of the concrete substrate / mm", value=300.0, step=10.0,
                         help="Width of the concrete substrate")
    fc = st.number_input("**fc'** - Compressive strength of concrete / MPa", value=30.0, step=1.0,
                         help="Compressive strength of concrete")
    Ef = st.number_input("**Ef** - Elastic modulus of FRP sheet / GPa", value=150.0, step=5.0,
                         help="Elastic modulus of FRP sheet")

with col2:
    tf = st.number_input("**tf** - Thickness of FRP sheet / mm", value=5.0, step=0.5,
                         help="Thickness of FRP sheet")
    bf = st.number_input("**bf** - Width of FRP sheet / mm", value=100.0, step=10.0,
                         help="Width of FRP sheet")
    lf = st.number_input("**lf** - Bond length of FRP sheet / mm", value=500.0, step=20.0,
                         help="Bond length of FRP sheet")

# ========== 计算派生参数 ==========
Kf = Ef * tf
bf_bc = bf / bc
Af = bf * lf
Df = bf * tf / lf

# 显示计算出的派生参数（只读，便于用户确认）
st.info(f"**Derived features (auto‑computed):**  Kf = {Kf:.2f} GPa·mm  |  bf/bc = {bf_bc:.3f}  |  Af = {Af:.0f} mm²  |  Df = {Df:.3f}")

# 准备模型输入（10个特征，顺序必须与训练时一致）
input_data = pd.DataFrame({
    'bc': [bc],
    'fc': [fc],
    'Ef': [Ef],
    'tf': [tf],
    'bf': [bf],
    'lf': [lf],
    'Kf': [Kf],
    'bf/bc': [bf_bc],
    'Af': [Af],
    'Df': [Df]
})

st.markdown("---")
col_btn, col_res = st.columns([1, 3])
with col_btn:
    predict_btn = st.button("🔮 Start predicting", type="primary", use_container_width=True)

if predict_btn:
    # 检查所有10个参数是否超出范围
    out_of_range_params = check_parameters_out_of_range(bc, fc, Ef, tf, bf, lf, Kf, bf_bc, Af, Df)
    
    with st.spinner("Prediction in progress, please wait..."):
        prediction = model.predict(input_data)[0]
    
    # 计算工程应用推荐值（折减系数 0.838）
    reduction_factor = 0.838
    recommended_value = prediction * reduction_factor
    
    with col_res:
        st.success("### Prediction result")
        # 创建两列并列显示
        col_pred, col_rec = st.columns(2)
        with col_pred:
            st.metric(label="📐 Bond strength $P_u$ (kN) - Model prediction", value=f"{prediction:.2f} kN")
        with col_rec:
            # 自定义 HTML 卡片，完全控制颜色和样式
            st.markdown(
                f"""
                <div style="background-color: #f0f2f6; border-radius: 0.5rem; padding: 1rem;">
                    <p style="font-size: {font_size}px; margin-bottom: 0.5rem; color: #333;">🛠️ Engineering application recommended value</p>
                    <p style="font-size: {font_size+24}px; font-weight: bold; color: red; margin: 0;">{recommended_value:.2f} kN</p>
                    <p style="font-size: {font_size-2}px; margin-top: 0.5rem; color: #666;">Reduction factor: {reduction_factor}</p>
                </div>
                """,
                unsafe_allow_html=True
            )
    
    if out_of_range_params:
       
        st.warning(f"⚠️ **The input parameters are outside the model's range. Use the prediction results with caution**\n\nThe following parameters are outside the range of the model：\n- " + "\n- ".join(out_of_range_params))
        
        with st.expander("📊 Please check the range of model parameters."):
            # 更新为10个参数的范围表格
            range_df = pd.DataFrame([
                {"parameters": "bc (mm) - Width of the concrete substrate", "range": f"{PARAM_RANGES['bc']['min']} – {PARAM_RANGES['bc']['max']}"},
                {"parameters": "fc' (MPa) - Compressive strength of concrete", "range": f"{PARAM_RANGES['fc']['min']} – {PARAM_RANGES['fc']['max']}"},
                {"parameters": "Ef (GPa) - Elastic modulus of FRP sheet", "range": f"{PARAM_RANGES['Ef']['min']} – {PARAM_RANGES['Ef']['max']}"},
                {"parameters": "tf (mm) - Thickness of FRP sheet", "range": f"{PARAM_RANGES['tf']['min']} – {PARAM_RANGES['tf']['max']}"},
                {"parameters": "bf (mm) - Width of FRP sheet", "range": f"{PARAM_RANGES['bf']['min']} – {PARAM_RANGES['bf']['max']}"},
                {"parameters": "lf (mm) - Bond length of FRP sheet", "range": f"{PARAM_RANGES['lf']['min']} – {PARAM_RANGES['lf']['max']}"},
                {"parameters": "Kf = Ef·tf (GPa·mm) - Stiffness of FRP sheet", "range": f"{PARAM_RANGES['Kf']['min']} – {PARAM_RANGES['Kf']['max']}"},
                {"parameters": "bf/bc - Width ratio between FRP sheet and concrete substrate", "range": f"{PARAM_RANGES['bf_bc']['min']} – {PARAM_RANGES['bf_bc']['max']}"},
                {"parameters": "Af = bf·lf (mm²) - Bond area of FRP sheet", "range": f"{PARAM_RANGES['Af']['min']} – {PARAM_RANGES['Af']['max']}"},
                {"parameters": "Df = bf·tf/lf - Slenderness ratio of FRP sheet", "range": f"{PARAM_RANGES['Df']['min']} – {PARAM_RANGES['Df']['max']}"}
            ])
            st.table(range_df)