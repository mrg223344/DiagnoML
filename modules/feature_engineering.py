import streamlit as st
import pandas as pd
import numpy as np
from sklearn.feature_selection import SelectKBest, f_classif, mutual_info_classif, RFE
from sklearn.ensemble import RandomForestClassifier
import matplotlib.pyplot as plt
import seaborn as sns

def feature_engineering_page():
    st.title("特征工程")
    
    st.markdown("""
    ### 特征工程模块
    
    本模块提供特征选择功能，帮助您从大量基因中筛选出最相关的特征，用于后续建模。
    
    支持的特征选择方法：
    
    1. **单变量特征选择**：基于统计测试选择最相关的特征
    2. **递归特征消除(RFE)**：递归地移除最不重要的特征
    3. **基于模型的特征选择**：使用随机森林等模型评估特征重要性
    
    请上传数据或使用示例数据进行操作。
    """)
    
    # 检查是否有数据加载
    if 'data' not in st.session_state:
        st.warning("请先在侧边栏上传数据或生成示例数据")
        return
    
    # 获取数据
    if 'train_data' in st.session_state:
        # 如果已经拆分了数据，使用训练集
        data = st.session_state['train_data']
        st.info("使用已拆分的训练集进行特征工程")
    else:
        # 否则使用全部数据
        data = st.session_state['data']
        st.info("使用全部数据进行特征工程")
    
    # 分离基因表达数据和临床数据
    # 假设所有以GENE_开头的列是基因表达数据
    gene_cols = [col for col in data.columns if col.startswith('GENE_')]
    clinical_cols = [col for col in data.columns if col not in gene_cols]
    
    if len(gene_cols) == 0:
        st.error("未检测到基因表达数据列（以GENE_开头的列）")
        return
    
    # 显示数据基本信息
    st.subheader("数据基本信息")
    st.write(f"样本数量: {data.shape[0]}")
    st.write(f"基因特征数量: {len(gene_cols)}")
    
    # 选择目标变量
    st.subheader("选择目标变量")
    target_col = st.selectbox(
        "选择目标变量",
        clinical_cols,
        index=clinical_cols.index('disease_status') if 'disease_status' in clinical_cols else 0,
        help="选择用于特征选择的目标变量，通常为分类变量或连续变量"
    )
    
    # 检查目标变量类型
    is_classification = pd.api.types.is_categorical_dtype(data[target_col]) or data[target_col].nunique() < 10
    
    # 特征选择方法
    st.subheader("特征选择方法")
    
    # 使用选项卡组织不同的特征选择方法
    tab1, tab2, tab3 = st.tabs(["单变量特征选择", "递归特征消除(RFE)", "基于模型的特征选择"])
    
    # 单变量特征选择
    with tab1:
        st.markdown("### 单变量特征选择")
        
        # 选择统计测试方法
        if is_classification:
            test_method = st.selectbox(
                "选择统计测试方法",
                ["ANOVA F-value", "互信息"],
                index=0,
                help="ANOVA适用于分类问题，互信息可用于分类和回归"
            )
        else:
            test_method = "互信息"
            st.info("对于连续目标变量，使用互信息作为统计测试方法")
        
        # 选择特征数量
        k_features = st.slider(
            "选择要保留的特征数量",
            min_value=5,
            max_value=min(50, len(gene_cols)),
            value=20,
            step=5,
            help="选择要保留的最相关特征的数量"
        )
        
        if st.button("执行单变量特征选择"):
            with st.spinner("正在进行特征选择..."):
                # 准备特征和目标变量
                X = data[gene_cols]
                y = data[target_col]
                
                # 根据选择的方法进行特征选择
                if test_method == "ANOVA F-value":
                    selector = SelectKBest(f_classif, k=k_features)
                else:  # 互信息
                    selector = SelectKBest(mutual_info_classif, k=k_features)
                
                # 拟合选择器
                selector.fit(X, y)
                
                # 获取选择的特征
                selected_features_mask = selector.get_support()
                selected_features = [gene_cols[i] for i in range(len(gene_cols)) if selected_features_mask[i]]
                
                # 获取特征得分
                feature_scores = selector.scores_
                
                # 创建特征重要性数据框
                feature_importance_df = pd.DataFrame({
                    'Feature': gene_cols,
                    'Score': feature_scores
                })
                
                # 按得分排序
                feature_importance_df = feature_importance_df.sort_values('Score', ascending=False)
                
                # 保存选择的特征
                st.session_state['selected_features'] = selected_features
                
                # 显示选择的特征
                st.success(f"特征选择完成! 已选择 {len(selected_features)} 个特征")
                
                # 显示特征重要性
                st.subheader("特征重要性")
                
                # 显示前k个特征的重要性
                top_features = feature_importance_df.head(k_features)
                
                # 绘制特征重要性条形图
                fig, ax = plt.subplots(figsize=(10, 6))
                sns.barplot(x='Score', y='Feature', data=top_features, ax=ax)
                ax.set_title(f"Top {k_features} 特征重要性")
                st.pyplot(fig)
                
                # 显示选择的特征列表
                st.subheader("选择的特征列表")
                st.dataframe(top_features)
                
                # 创建包含选择特征的数据集
                selected_data = data[selected_features + [target_col]]
                
                # 保存到会话状态
                st.session_state['feature_selected_data'] = selected_data
                
                # 显示选择特征后的数据预览
                st.subheader("选择特征后的数据预览")
                st.dataframe(selected_data.head())
    
    # 递归特征消除(RFE)
    with tab2:
        st.markdown("### 递归特征消除(RFE)")
        
        # 选择特征数量
        rfe_k_features = st.slider(
            "选择要保留的特征数量 (RFE)",
            min_value=5,
            max_value=min(50, len(gene_cols)),
            value=20,
            step=5,
            help="选择要保留的最重要特征的数量"
        )
        
        # 选择步长
        step = st.slider(
            "选择每次消除的特征数量",
            min_value=1,
            max_value=10,
            value=1,
            help="每次迭代中要移除的特征数量"
        )
        
        if st.button("执行递归特征消除"):
            with st.spinner("正在进行递归特征消除..."):
                # 准备特征和目标变量
                X = data[gene_cols]
                y = data[target_col]
                
                # 创建基础模型
                if is_classification:
                    estimator = RandomForestClassifier(n_estimators=100, random_state=42)
                else:
                    from sklearn.ensemble import RandomForestRegressor
                    estimator = RandomForestRegressor(n_estimators=100, random_state=42)
                
                # 创建RFE选择器
                rfe = RFE(estimator=estimator, n_features_to_select=rfe_k_features, step=step)
                
                # 拟合选择器
                rfe.fit(X, y)
                
                # 获取选择的特征
                selected_features_mask = rfe.support_
                selected_features = [gene_cols[i] for i in range(len(gene_cols)) if selected_features_mask[i]]
                
                # 获取特征排名
                feature_ranking = rfe.ranking_
                
                # 创建特征重要性数据框
                feature_importance_df = pd.DataFrame({
                    'Feature': gene_cols,
                    'Ranking': feature_ranking
                })
                
                # 按排名排序
                feature_importance_df = feature_importance_df.sort_values('Ranking')
                
                # 保存选择的特征
                st.session_state['selected_features'] = selected_features
                
                # 显示选择的特征
                st.success(f"递归特征消除完成! 已选择 {len(selected_features)} 个特征")
                
                # 显示特征排名
                st.subheader("特征排名")
                
                # 显示选择的特征
                selected_features_df = feature_importance_df[feature_importance_df['Ranking'] == 1]
                
                # 绘制特征排名条形图
                fig, ax = plt.subplots(figsize=(10, 6))
                sns.barplot(x='Ranking', y='Feature', data=selected_features_df, ax=ax)
                ax.set_title(f"选择的 {rfe_k_features} 个特征")
                st.pyplot(fig)
                
                # 显示选择的特征列表
                st.subheader("选择的特征列表")
                st.dataframe(selected_features_df)
                
                # 创建包含选择特征的数据集
                selected_data = data[selected_features + [target_col]]
                
                # 保存到会话状态
                st.session_state['feature_selected_data'] = selected_data
                
                # 显示选择特征后的数据预览
                st.subheader("选择特征后的数据预览")
                st.dataframe(selected_data.head())
    
    # 基于模型的特征选择
    with tab3:
        st.markdown("### 基于模型的特征选择")
        
        # 选择模型
        model_type = st.selectbox(
            "选择模型类型",
            ["随机森林"],
            index=0
        )
        
        # 选择特征数量
        model_k_features = st.slider(
            "选择要保留的特征数量 (基于模型)",
            min_value=5,
            max_value=min(50, len(gene_cols)),
            value=20,
            step=5,
            help="选择要保留的最重要特征的数量"
        )
        
        if st.button("执行基于模型的特征选择"):
            with st.spinner("正在进行基于模型的特征选择..."):
                # 准备特征和目标变量
                X = data[gene_cols]
                y = data[target_col]
                
                # 创建模型
                if model_type == "随机森林":
                    if is_classification:
                        model = RandomForestClassifier(n_estimators=100, random_state=42)
                    else:
                        from sklearn.ensemble import RandomForestRegressor
                        model = RandomForestRegressor(n_estimators=100, random_state=42)
                
                # 拟合模型
                model.fit(X, y)
                
                # 获取特征重要性
                feature_importances = model.feature_importances_
                
                # 创建特征重要性数据框
                feature_importance_df = pd.DataFrame({
                    'Feature': gene_cols,
                    'Importance': feature_importances
                })
                
                # 按重要性排序
                feature_importance_df = feature_importance_df.sort_values('Importance', ascending=False)
                
                # 选择前k个特征
                selected_features = feature_importance_df.head(model_k_features)['Feature'].tolist()
                
                # 保存选择的特征
                st.session_state['selected_features'] = selected_features
                
                # 显示选择的特征
                st.success(f"基于模型的特征选择完成! 已选择 {len(selected_features)} 个特征")
                
                # 显示特征重要性
                st.subheader("特征重要性")
                
                # 显示前k个特征的重要性
                top_features = feature_importance_df.head(model_k_features)
                
                # 绘制特征重要性条形图
                fig, ax = plt.subplots(figsize=(10, 6))
                sns.barplot(x='Importance', y='Feature', data=top_features, ax=ax)
                ax.set_title(f"Top {model_k_features} 特征重要性")
                st.pyplot(fig)
                
                # 显示选择的特征列表
                st.subheader("选择的特征列表")
                st.dataframe(top_features)
                
                # 创建包含选择特征的数据集
                selected_data = data[selected_features + [target_col]]
                
                # 保存到会话状态
                st.session_state['feature_selected_data'] = selected_data
                
                # 显示选择特征后的数据预览
                st.subheader("选择特征后的数据预览")
                st.dataframe(selected_data.head())
    
    # 添加技术说明
    with st.expander("技术说明"):
        st.markdown("""
        #### 特征选择的重要性
        
        在RNA-seq数据分析中，特征选择是一个关键步骤，因为：
        
        1. **降低维度**：RNA-seq数据通常包含成千上万个基因，而样本数量相对较少，这会导致"维度灾难"
        2. **减少过拟合**：减少特征数量可以降低模型过拟合的风险
        3. **提高解释性**：选择最相关的基因有助于理解生物学机制
        4. **提高计算效率**：减少特征数量可以加快模型训练和预测
        
        #### 特征选择方法
        
        1. **单变量特征选择**：
           - 对每个特征单独进行统计测试，评估其与目标变量的相关性
           - ANOVA F-value适用于分类问题，评估特征在不同类别间的区分能力
           - 互信息可用于分类和回归问题，能够捕捉非线性关系
        
        2. **递归特征消除(RFE)**：
           - 首先使用所有特征训练模型
           - 根据特征重要性，递归地移除最不重要的特征
           - 最终保留指定数量的最重要特征
        
        3. **基于模型的特征选择**：
           - 使用模型（如随机森林）的内置特征重要性评估机制
           - 随机森林可以评估特征在决策树中的平均不纯度减少
           - 这种方法可以捕捉特征之间的交互作用
        """)

if __name__ == "__main__":
    feature_engineering_page()
