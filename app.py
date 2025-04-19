import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import time
import os
import sys

# 获取当前脚本所在的目录
current_dir = os.path.dirname(os.path.abspath(__file__))
# 构建utils目录的路径
utils_dir = os.path.join(current_dir, 'utils')
# 将utils目录添加到模块搜索路径
sys.path.append(utils_dir)

# 导入各个模块
from modules.expression_correction import expression_correction_page
from modules.data_splitting import data_splitting_page
from modules.data_preprocessing import data_preprocessing_page
from modules.feature_engineering import feature_engineering_page
from modules.data_balancing import data_balancing_page
from modules.statistical_matching import statistical_matching_page
from modules.machine_learning import machine_learning_page
from modules.data_visualization import data_visualization_page
from modules.model_interpretation import model_interpretation_page
from modules.model_validation import model_validation_page
from modules.enrichment_analysis import enrichment_analysis_page
from utils.data_generator import generate_rna_seq_data

# 设置页面配置
st.set_page_config(
    page_title="DiagnoML——实体肿瘤生物标志物和预测平台",
    page_icon="🧬",
    layout="wide",
    initial_sidebar_state="expanded"
)

# 自定义CSS样式
def load_css():
    st.markdown("""
    <style>
    .main {
        background-color: #f8f9fa;
    }
    .stApp {
        max-width: 1200px;
        margin: 0 auto;
    }
    h1, h2, h3 {
        color: #2c3e50;
    }
    .stSidebar {
        background-color: #f1f3f5;
    }
    .stButton>button {
        background-color: #4e73df;
        color: white;
    }
    .stButton>button:hover {
        background-color: #2e59d9;
        color: white;
    }
    .info-box {
        padding: 1rem;
        border-radius: 0.5rem;
        background-color: #e3f2fd;
        border-left: 5px solid #2196f3;
        margin-bottom: 1rem;
    }
    .success-box {
        padding: 1rem;
        border-radius: 0.5rem;
        background-color: #e8f5e9;
        border-left: 5px solid #4caf50;
        margin-bottom: 1rem;
    }
    .warning-box {
        padding: 1rem;
        border-radius: 0.5rem;
        background-color: #fff8e1;
        border-left: 5px solid #ff9800;
        margin-bottom: 1rem;
    }
    </style>
    """, unsafe_allow_html=True)

# 加载CSS样式
load_css()

# 创建侧边栏
def sidebar():
    with st.sidebar:
        st.image("https://cdn-icons-png.flaticon.com/512/2103/2103633.png", width=100)
        st.title("RNA-seq自动机器学习平台")
        st.markdown("---")

        # 数据加载部分
        st.header("数据加载")

        data_option = st.radio(
            "选择数据来源",
            ["上传数据", "生成示例数据"],
            index=1
        )

        if data_option == "上传数据":
            uploaded_file = st.file_uploader("上传RNA-seq数据 (CSV格式)", type=["csv"])

            if uploaded_file is not None:
                try:
                    data = pd.read_csv(uploaded_file)
                    st.success(f"成功加载数据: {uploaded_file.name}")
                    st.session_state['data'] = data
                    st.session_state['data_source'] = "uploaded"

                    # 显示数据基本信息
                    st.write(f"样本数: {data.shape[0]}")
                    st.write(f"特征数: {data.shape[1]}")
                except Exception as e:
                    st.error(f"加载数据时出错: {e}")

        elif data_option == "生成示例数据":
            st.write("生成RNA-seq示例数据")

            n_samples = st.slider("样本数量", 50, 500, 100)
            n_genes = st.slider("基因数量", 50, 1000, 200)
            n_clinical = st.slider("临床特征数量", 3, 20, 5)

            if st.button("生成示例数据"):
                with st.spinner("正在生成示例数据..."):
                    data = generate_rna_seq_data(n_samples, n_genes, n_clinical)
                    st.session_state['data'] = data
                    st.session_state['data_source'] = "generated"

                    # 显示数据基本信息
                    st.success("成功生成示例数据")
                    st.write(f"样本数: {data.shape[0]}")
                    st.write(f"特征数: {data.shape[1]}")

        st.markdown("---")

        # 功能导航
        st.header("功能导航")

        page = st.radio(
            "选择功能模块",
            ["首页",
             "表达量校正",
             "数据拆分",
             "数据预处理",
             "特征工程",
             "数据平衡",
             "统计匹配",
             "机器学习模型",
             "数据可视化",
             "可解释性分析",
             "模型验证",
             "富集分析"],
            index=0
        )

        st.markdown("---")

        # 关于信息
        st.markdown("### 关于")
        st.markdown("RNA-seq自动机器学习平台 v1.0")
        st.markdown("© 2025 RNA-seq AutoML Team")

    return page

# 首页内容
def home_page():
    st.title("DiagnoML——实体肿瘤生物标志物和预测平台")

    st.markdown("""
    <div class="info-box">
    <h3>欢迎使用DiagnoML——实体肿瘤生物标志物和预测平台</h3>
    <p>本平台提供了一站式数据分析解决方案，集成了数据预处理、特征工程、机器学习建模和结果可视化等功能。</p>
    </div>
    """, unsafe_allow_html=True)

    # 平台介绍
    st.header("平台介绍")

    st.markdown("""
    DiagnoML——实体肿瘤生物标志物和预测平台是一个专为RNA-seq数据分析设计的综合性工具，旨在帮助研究人员快速、高效地从RNA-seq数据中提取有价值的信息和构建预测模型。
    
    本平台具有以下特点：
    
    1. **用户友好**：简洁直观的界面设计，无需编程经验即可完成复杂分析
    2. **全流程支持**：覆盖从数据预处理到模型构建的完整分析流程
    3. **自动化建模**：内置自动特征选择和模型优化算法
    4. **可视化分析**：丰富的可视化工具，帮助理解数据和解释结果
    5. **可解释性**：提供模型可解释性分析，揭示基因与表型的关系
    """)

    # 功能模块介绍
    st.header("功能模块")

    col1, col2 = st.columns(2)

    with col1:
        st.subheader("数据处理")
        st.markdown("""
        - **表达量校正**：TPM转换和z-score标准化
        - **数据拆分**：训练集和测试集划分
        - **数据预处理**：缺失值处理、编码转换、标准化
        - **特征工程**：特征选择、降维、特征变换
        - **数据平衡**：处理类别不平衡问题
        - **统计匹配**：匹配样本特征，减少混杂因素影响
        """)

    with col2:
        st.subheader("分析与建模")
        st.markdown("""
        - **机器学习模型**：自定义建模和自动化建模
        - **数据可视化**：探索性数据分析和结果可视化
        - **可解释性分析**：特征重要性和SHAP值分析
        - **模型验证**：性能评估和决策曲线分析
        - **富集分析**：GO、KEGG通路和基因集富集分析
        """)

    # 使用流程
    st.header("使用流程")

    st.markdown("""
    1. **数据加载**：上传RNA-seq数据或生成示例数据
    2. **数据预处理**：进行表达量校正、数据拆分和预处理
    3. **特征工程**：选择重要特征，优化特征表示
    4. **模型构建**：选择合适的机器学习算法构建模型
    5. **模型评估**：验证模型性能，分析模型可解释性
    6. **结果解读**：通过富集分析理解生物学意义
    """)

    # 检查是否已加载数据
    if 'data' in st.session_state:
        st.markdown("""
        <div class="success-box">
        <h3>数据已加载</h3>
        <p>您已成功加载数据，可以开始分析流程。请从侧边栏选择功能模块。</p>
        </div>
        """, unsafe_allow_html=True)
    else:
        st.markdown("""
        <div class="warning-box">
        <h3>未检测到数据</h3>
        <p>请先在侧边栏上传数据或生成示例数据。</p>
        </div>
        """, unsafe_allow_html=True)

# 主函数
def main():
    # 显示侧边栏并获取选择的页面
    page = sidebar()

    # 根据选择的页面显示相应内容
    if page == "首页":
        home_page()
    elif page == "表达量校正":
        expression_correction_page()
    elif page == "数据拆分":
        data_splitting_page()
    elif page == "数据预处理":
        data_preprocessing_page()
    elif page == "特征工程":
        feature_engineering_page()
    elif page == "数据平衡":
        data_balancing_page()
    elif page == "统计匹配":
        statistical_matching_page()
    elif page == "机器学习模型":
        machine_learning_page()
    elif page == "数据可视化":
        data_visualization_page()
    elif page == "可解释性分析":
        model_interpretation_page()
    elif page == "模型验证":
        model_validation_page()
    elif page == "富集分析":
        enrichment_analysis_page()

if __name__ == "__main__":
    main()
