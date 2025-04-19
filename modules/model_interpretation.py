import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.inspection import permutation_importance
import shap
import time

def model_interpretation_page():
    st.title("可解释性分析")
    
    st.markdown("""
    ### 可解释性分析模块
    
    本模块提供机器学习模型的可解释性分析功能，帮助您理解模型的决策机制和特征的重要性。
    
    支持的可解释性方法：
    
    1. **特征重要性**：评估每个特征对模型预测的贡献
    2. **SHAP值分析**：基于博弈论的方法，解释每个特征对每个样本预测的贡献
    3. **部分依赖图**：展示特征值变化对模型预测的影响
    
    请先训练机器学习模型，然后使用本模块进行可解释性分析。
    """)
    
    # 检查是否有数据加载
    if 'data' not in st.session_state:
        st.warning("请先在侧边栏上传数据或生成示例数据")
        return
    
    # 检查是否已训练模型
    if 'model' not in st.session_state:
        st.warning("请先在机器学习模型模块训练模型")
        return
    
    # 获取模型和特征
    model = st.session_state['model']
    selected_features = st.session_state['model_features']
    
    # 获取数据
    if 'test_data' in st.session_state:
        # 使用测试集进行解释
        data = st.session_state['test_data']
        st.info("使用测试集数据进行模型解释")
    else:
        # 使用全部数据
        data = st.session_state['data']
        st.info("使用全部数据进行模型解释")
    
    # 准备特征和目标变量
    # 分离基因表达数据和临床数据
    gene_cols = [col for col in data.columns if col.startswith('GENE_')]
    clinical_cols = [col for col in data.columns if col not in gene_cols]
    
    # 选择目标变量
    target_col = st.selectbox(
        "选择目标变量",
        clinical_cols,
        index=clinical_cols.index('disease_status') if 'disease_status' in clinical_cols else 0,
        help="选择用于解释的目标变量，通常为分类变量"
    )
    
    # 准备特征和目标变量
    X = data[selected_features]
    y = data[target_col]
    
    # 选择可解释性方法
    st.subheader("选择可解释性方法")
    
    interpretation_method = st.radio(
        "可解释性方法",
        ["特征重要性", "SHAP值分析", "部分依赖图"],
        index=0
    )
    
    # 特征重要性
    if interpretation_method == "特征重要性":
        st.markdown("### 特征重要性分析")
        
        # 选择特征重要性计算方法
        importance_method = st.radio(
            "特征重要性计算方法",
            ["模型内置重要性", "排列重要性"],
            index=0
        )
        
        if st.button("计算特征重要性"):
            with st.spinner("正在计算特征重要性..."):
                if importance_method == "模型内置重要性":
                    # 检查模型是否支持内置特征重要性
                    if hasattr(model[-1], 'feature_importances_'):
                        # 对于随机森林、梯度提升树等
                        importances = model[-1].feature_importances_
                        importance_type = "特征重要性"
                    elif hasattr(model[-1], 'coef_'):
                        # 对于线性模型
                        importances = np.abs(model[-1].coef_[0])
                        importance_type = "系数绝对值"
                    else:
                        st.error("当前模型不支持内置特征重要性计算，请尝试排列重要性")
                        return
                
                elif importance_method == "排列重要性":
                    # 使用排列重要性
                    perm_importance = permutation_importance(model, X, y, n_repeats=10, random_state=42)
                    importances = perm_importance.importances_mean
                    importance_type = "排列重要性"
                
                # 创建特征重要性数据框
                feature_importance_df = pd.DataFrame({
                    'Feature': selected_features,
                    'Importance': importances
                })
                
                # 按重要性排序
                feature_importance_df = feature_importance_df.sort_values('Importance', ascending=False)
                
                # 绘制特征重要性条形图
                fig, ax = plt.subplots(figsize=(10, 8))
                sns.barplot(x='Importance', y='Feature', data=feature_importance_df, ax=ax)
                ax.set_title(f"特征重要性 ({importance_type})")
                st.pyplot(fig)
                
                # 显示特征重要性表格
                st.subheader("特征重要性表格")
                st.dataframe(feature_importance_df)
    
    # SHAP值分析
    elif interpretation_method == "SHAP值分析":
        st.markdown("### SHAP值分析")
        
        st.info("SHAP (SHapley Additive exPlanations) 是一种基于博弈论的方法，用于解释每个特征对每个样本预测的贡献。")
        
        # 选择SHAP可视化类型
        shap_viz_type = st.radio(
            "SHAP可视化类型",
            ["摘要图", "依赖图", "力图"],
            index=0
        )
        
        # 选择样本数量
        n_samples = st.slider("选择用于SHAP计算的样本数量", 50, 200, 100, 
                             help="较大的样本数量会提高准确性，但会增加计算时间")
        
        if st.button("计算SHAP值"):
            with st.spinner("正在计算SHAP值..."):
                # 随机选择样本
                if len(X) > n_samples:
                    sample_indices = np.random.choice(len(X), n_samples, replace=False)
                    X_sample = X.iloc[sample_indices]
                else:
                    X_sample = X
                
                try:
                    # 创建SHAP解释器
                    explainer = shap.Explainer(model[-1], X_sample)
                    
                    # 计算SHAP值
                    shap_values = explainer(X_sample)
                    
                    # 根据选择的可视化类型创建图表
                    if shap_viz_type == "摘要图":
                        # 创建摘要图
                        plt.figure(figsize=(10, 8))
                        shap.summary_plot(shap_values, X_sample, plot_type="bar")
                        st.pyplot(plt.gcf())
                        
                        plt.figure(figsize=(10, 8))
                        shap.summary_plot(shap_values, X_sample)
                        st.pyplot(plt.gcf())
                    
                    elif shap_viz_type == "依赖图":
                        # 选择要可视化的特征
                        feature_to_plot = st.selectbox(
                            "选择要可视化的特征",
                            selected_features
                        )
                        
                        # 创建依赖图
                        plt.figure(figsize=(10, 8))
                        shap.dependence_plot(
                            feature_to_plot, 
                            shap_values.values, 
                            X_sample,
                            feature_names=selected_features
                        )
                        st.pyplot(plt.gcf())
                    
                    elif shap_viz_type == "力图":
                        # 选择要可视化的样本
                        sample_index = st.slider("选择要可视化的样本索引", 0, len(X_sample)-1, 0)
                        
                        # 创建力图
                        plt.figure(figsize=(10, 8))
                        shap.force_plot(
                            explainer.expected_value, 
                            shap_values.values[sample_index,:], 
                            X_sample.iloc[sample_index,:],
                            feature_names=selected_features,
                            matplotlib=True
                        )
                        st.pyplot(plt.gcf())
                
                except Exception as e:
                    st.error(f"计算SHAP值时出错: {e}")
                    st.info("SHAP值计算可能不支持当前模型类型，请尝试其他可解释性方法")
    
    # 部分依赖图
    elif interpretation_method == "部分依赖图":
        st.markdown("### 部分依赖图")
        
        st.info("部分依赖图展示了特征值变化对模型预测的影响，帮助理解特征与目标变量之间的关系。")
        
        # 选择要可视化的特征
        feature_to_plot = st.selectbox(
            "选择要可视化的特征",
            selected_features
        )
        
        if st.button("生成部分依赖图"):
            with st.spinner("正在生成部分依赖图..."):
                try:
                    from sklearn.inspection import partial_dependence, plot_partial_dependence
                    
                    # 计算部分依赖
                    pdp_result = partial_dependence(
                        model, X, [selected_features.index(feature_to_plot)]
                    )
                    
                    # 提取结果
                    pdp_values = pdp_result["average"]
                    pdp_x = pdp_result["values"][0]
                    
                    # 绘制部分依赖图
                    fig, ax = plt.subplots(figsize=(10, 6))
                    ax.plot(pdp_x, pdp_values[0])
                    ax.set_xlabel(feature_to_plot)
                    ax.set_ylabel("部分依赖")
                    ax.set_title(f"{feature_to_plot} 的部分依赖图")
                    ax.grid(True)
                    st.pyplot(fig)
                    
                    # 显示部分依赖值
                    pdp_df = pd.DataFrame({
                        feature_to_plot: pdp_x,
                        "部分依赖": pdp_values[0]
                    })
                    st.subheader("部分依赖值")
                    st.dataframe(pdp_df)
                
                except Exception as e:
                    st.error(f"生成部分依赖图时出错: {e}")
                    st.info("部分依赖图计算可能不支持当前模型类型，请尝试其他可解释性方法")
    
    # 添加技术说明
    with st.expander("技术说明"):
        st.markdown("""
        #### 模型可解释性的重要性
        
        在生物医学研究中，模型的可解释性与预测性能同样重要。可解释性分析可以：
        
        1. **揭示生物学机制**：识别与疾病或表型相关的关键基因
        2. **验证已知知识**：确认模型是否捕捉到已知的生物学关系
        3. **发现新知识**：发现潜在的新生物标志物或治疗靶点
        4. **增强可信度**：提高研究人员和临床医生对模型的信任
        
        #### 可解释性方法
        
        1. **特征重要性**：
           - 模型内置重要性：基于模型的内部机制计算特征重要性，如随机森林中的不纯度减少
           - 排列重要性：通过随机打乱特征值并观察模型性能变化来评估特征重要性
        
        2. **SHAP值分析**：
           - 基于博弈论的Shapley值，计算每个特征对每个样本预测的贡献
           - 摘要图：展示所有特征的整体重要性和影响方向
           - 依赖图：展示特征值与其SHAP值之间的关系
           - 力图：展示单个样本的预测解释
        
        3. **部分依赖图**：
           - 展示特征值变化对模型预测的平均影响
           - 帮助理解特征与目标变量之间的关系，包括线性、非线性或阈值效应
        
        #### 结果解读
        
        - **正SHAP值**：表示该特征增加了预测为正类的概率
        - **负SHAP值**：表示该特征降低了预测为正类的概率
        - **特征重要性高**：表示该特征对模型预测有显著影响
        - **部分依赖图斜率**：表示特征值变化对预测的影响程度和方向
        """)

if __name__ == "__main__":
    model_interpretation_page()
