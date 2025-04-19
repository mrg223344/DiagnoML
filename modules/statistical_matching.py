import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import StandardScaler
from sklearn.neighbors import NearestNeighbors
from sklearn.linear_model import LogisticRegression

def statistical_matching_page():
    st.title("统计匹配")
    
    st.markdown("""
    ### 统计匹配模块
    
    本模块提供倾向得分匹配分析功能，用于减少混杂因素的影响，使处理组和对照组在协变量分布上更加平衡。
    
    主要功能：
    
    1. **倾向得分计算**：使用逻辑回归等方法估计样本属于处理组的概率
    2. **匹配方法**：支持最近邻匹配、半径匹配等方法
    3. **平衡评估**：评估匹配前后协变量的平衡性
    
    请上传数据或使用示例数据进行操作。
    """)
    
    # 检查是否有数据加载
    if 'data' not in st.session_state:
        st.warning("请先在侧边栏上传数据或生成示例数据")
        return
    
    # 获取数据
    data = st.session_state['data']
    
    # 分离基因表达数据和临床数据
    # 假设所有以GENE_开头的列是基因表达数据
    gene_cols = [col for col in data.columns if col.startswith('GENE_')]
    clinical_cols = [col for col in data.columns if col not in gene_cols]
    
    # 选择处理变量
    st.subheader("选择处理变量")
    
    # 筛选二分类变量
    binary_cols = [col for col in clinical_cols 
                  if pd.api.types.is_categorical_dtype(data[col]) or 
                  pd.api.types.is_object_dtype(data[col]) or
                  (pd.api.types.is_numeric_dtype(data[col]) and data[col].nunique() == 2)]
    
    if not binary_cols:
        st.error("未检测到合适的二分类变量，统计匹配需要一个二分类的处理变量")
        return
    
    treatment_col = st.selectbox(
        "选择处理变量",
        binary_cols,
        index=binary_cols.index('disease_status') if 'disease_status' in binary_cols else 0,
        help="选择表示处理/对照组的二分类变量"
    )
    
    # 确保处理变量是二分类的
    unique_values = data[treatment_col].unique()
    if len(unique_values) != 2:
        st.error(f"处理变量 {treatment_col} 不是二分类变量，请选择另一个变量")
        return
    
    # 显示处理变量的分布
    st.subheader("处理变量分布")
    
    # 计算处理变量分布
    treatment_counts = data[treatment_col].value_counts()
    treatment_percentages = data[treatment_col].value_counts(normalize=True) * 100
    
    # 创建分布数据框
    distribution_df = pd.DataFrame({
        '类别': treatment_counts.index,
        '样本数': treatment_counts.values,
        '百分比': treatment_percentages.values.round(2)
    })
    
    # 显示分布表格
    st.dataframe(distribution_df)
    
    # 绘制分布图
    fig, ax = plt.subplots(figsize=(10, 6))
    sns.barplot(x=treatment_counts.index.astype(str), y=treatment_counts.values, ax=ax)
    ax.set_title(f"{treatment_col} 分布")
    ax.set_xlabel("类别")
    ax.set_ylabel("样本数")
    
    # 在柱状图上添加数值标签
    for i, v in enumerate(treatment_counts.values):
        ax.text(i, v + 0.1, str(v), ha='center')
    
    st.pyplot(fig)
    
    # 选择协变量
    st.subheader("选择协变量")
    
    # 筛选可能的协变量（排除处理变量和非数值变量）
    possible_covariates = [col for col in clinical_cols 
                          if col != treatment_col and 
                          pd.api.types.is_numeric_dtype(data[col])]
    
    if not possible_covariates:
        st.error("未检测到合适的数值型协变量，请确保数据中包含数值型临床特征")
        return
    
    selected_covariates = st.multiselect(
        "选择用于匹配的协变量",
        possible_covariates,
        default=possible_covariates[:min(5, len(possible_covariates))],
        help="选择可能影响处理分配的协变量，这些变量将用于计算倾向得分"
    )
    
    if not selected_covariates:
        st.warning("请至少选择一个协变量")
        return
    
    # 匹配设置
    st.subheader("匹配设置")
    
    # 选择匹配方法
    matching_method = st.selectbox(
        "匹配方法",
        ["最近邻匹配", "半径匹配"],
        index=0,
        help="最近邻匹配为每个处理组样本找到最相似的对照组样本；半径匹配在指定距离内寻找所有匹配样本"
    )
    
    # 根据匹配方法设置参数
    if matching_method == "最近邻匹配":
        n_neighbors = st.slider(
            "每个处理组样本匹配的对照组样本数",
            1, 5, 1,
            help="为每个处理组样本匹配的对照组样本数量"
        )
        
        caliper = None
    
    elif matching_method == "半径匹配":
        caliper = st.slider(
            "匹配半径（倾向得分标准差的倍数）",
            0.1, 1.0, 0.2, 0.1,
            help="匹配半径，表示为倾向得分标准差的倍数"
        )
        
        n_neighbors = None
    
    # 是否允许替换
    replacement = st.checkbox(
        "允许替换",
        value=False,
        help="如果勾选，对照组样本可以被多次匹配；否则每个对照组样本最多被匹配一次"
    )
    
    # 执行匹配
    if st.button("执行统计匹配"):
        with st.spinner("正在进行统计匹配..."):
            # 准备数据
            # 确保处理变量是二进制的（0/1）
            treatment_values = data[treatment_col].unique()
            treatment_map = {treatment_values[0]: 0, treatment_values[1]: 1}
            treatment = data[treatment_col].map(treatment_map)
            
            # 准备协变量
            covariates = data[selected_covariates].copy()
            
            # 处理缺失值
            if covariates.isnull().any().any():
                st.warning("协变量中存在缺失值，将使用均值填充")
                covariates = covariates.fillna(covariates.mean())
            
            # 标准化协变量
            scaler = StandardScaler()
            covariates_scaled = scaler.fit_transform(covariates)
            covariates_scaled_df = pd.DataFrame(covariates_scaled, columns=selected_covariates)
            
            # 计算倾向得分
            try:
                # 使用逻辑回归估计倾向得分
                ps_model = LogisticRegression(random_state=42)
                ps_model.fit(covariates_scaled, treatment)
                
                # 获取倾向得分
                propensity_scores = ps_model.predict_proba(covariates_scaled)[:, 1]
                
                # 添加倾向得分到数据
                data_with_ps = data.copy()
                data_with_ps['propensity_score'] = propensity_scores
                
                # 分离处理组和对照组
                treated = data_with_ps[treatment == 1]
                control = data_with_ps[treatment == 0]
                
                # 执行匹配
                if matching_method == "最近邻匹配":
                    # 使用最近邻算法进行匹配
                    nbrs = NearestNeighbors(n_neighbors=n_neighbors, algorithm='ball_tree').fit(
                        control['propensity_score'].values.reshape(-1, 1))
                    
                    # 为每个处理组样本找到最近的对照组样本
                    distances, indices = nbrs.kneighbors(treated['propensity_score'].values.reshape(-1, 1))
                    
                    # 获取匹配的对照组样本
                    matched_control_indices = []
                    for i, idx_array in enumerate(indices):
                        for idx in idx_array:
                            if not replacement and idx in matched_control_indices:
                                continue
                            matched_control_indices.append(control.iloc[idx].name)
                    
                    # 创建匹配后的数据集
                    matched_control = data_with_ps.loc[matched_control_indices]
                    matched_data = pd.concat([treated, matched_control])
                
                elif matching_method == "半径匹配":
                    # 计算倾向得分的标准差
                    ps_std = data_with_ps['propensity_score'].std()
                    radius = caliper * ps_std
                    
                    # 使用半径匹配
                    matched_control_indices = []
                    
                    for i, t_ps in enumerate(treated['propensity_score']):
                        # 找到在半径内的所有对照组样本
                        matches = control[abs(control['propensity_score'] - t_ps) <= radius]
                        
                        if not matches.empty:
                            if replacement:
                                # 如果允许替换，使用所有匹配的样本
                                matched_control_indices.extend(matches.index)
                            else:
                                # 如果不允许替换，只使用尚未匹配的样本
                                available_matches = [idx for idx in matches.index if idx not in matched_control_indices]
                                if available_matches:
                                    # 选择最接近的一个
                                    closest_match = min(available_matches, 
                                                       key=lambda x: abs(control.loc[x, 'propensity_score'] - t_ps))
                                    matched_control_indices.append(closest_match)
                    
                    # 创建匹配后的数据集
                    matched_control = data_with_ps.loc[matched_control_indices]
                    matched_data = pd.concat([treated, matched_control])
                
                # 保存匹配后的数据
                st.session_state['matched_data'] = matched_data
                
                # 显示匹配结果
                st.success(f"匹配完成! 匹配前: 处理组 {len(treated)}样本, 对照组 {len(control)}样本; 匹配后: 处理组 {len(treated)}样本, 对照组 {len(matched_control)}样本")
                
                # 评估匹配前后的平衡性
                st.subheader("匹配前后协变量平衡性评估")
                
                # 计算标准化均值差异
                balance_results = []
                
                for cov in selected_covariates:
                    # 匹配前
                    treated_mean_before = data[treatment == 1][cov].mean()
                    control_mean_before = data[treatment == 0][cov].mean()
                    pooled_std_before = np.sqrt((data[treatment == 1][cov].var() + data[treatment == 0][cov].var()) / 2)
                    
                    if pooled_std_before == 0:
                        std_diff_before = 0
                    else:
                        std_diff_before = (treated_mean_before - control_mean_before) / pooled_std_before
                    
                    # 匹配后
                    treated_mean_after = matched_data[matched_data[treatment_col] == treatment_values[1]][cov].mean()
                    control_mean_after = matched_data[matched_data[treatment_col] == treatment_values[0]][cov].mean()
                    pooled_std_after = np.sqrt((matched_data[matched_data[treatment_col] == treatment_values[1]][cov].var() + 
                                              matched_data[matched_data[treatment_col] == treatment_values[0]][cov].var()) / 2)
                    
                    if pooled_std_after == 0:
                        std_diff_after = 0
                    else:
                        std_diff_after = (treated_mean_after - control_mean_after) / pooled_std_after
                    
                    balance_results.append({
                        '协变量': cov,
                        '匹配前标准化均值差异': std_diff_before,
                        '匹配后标准化均值差异': std_diff_after,
                        '改善百分比': (1 - abs(std_diff_after) / abs(std_diff_before)) * 100 if std_diff_before != 0 else 0
                    })
                
                # 显示平衡性评估结果
                balance_df = pd.DataFrame(balance_results)
                st.dataframe(balance_df)
                
                # 绘制匹配前后的标准化均值差异
                fig, ax = plt.subplots(figsize=(12, 6))
                
                # 准备绘图数据
                plot_data = pd.melt(
                    balance_df, 
                    id_vars=['协变量'], 
                    value_vars=['匹配前标准化均值差异', '匹配后标准化均值差异'],
                    var_name='匹配状态',
                    value_name='标准化均值差异'
                )
                
                # 绘制条形图
                sns.barplot(x='协变量', y='标准化均值差异', hue='匹配状态', data=plot_data, ax=ax)
                ax.set_title("匹配前后协变量平衡性比较")
                ax.set_xlabel("协变量")
                ax.set_ylabel("标准化均值差异")
                ax.axhline(y=0, color='r', linestyle='-')
                ax.axhline(y=0.1, color='r', linestyle='--')
                ax.axhline(y=-0.1, color='r', linestyle='--')
                plt.xticks(rotation=45)
                plt.tight_layout()
                
                st.pyplot(fig)
                
                # 绘制倾向得分分布
                st.subheader("倾向得分分布")
                
                fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))
                
                # 匹配前
                sns.histplot(
                    data=data_with_ps, 
                    x='propensity_score', 
                    hue=treatment_col,
                    bins=20,
                    ax=ax1
                )
                ax1.set_title("匹配前倾向得分分布")
                ax1.set_xlabel("倾向得分")
                ax1.set_ylabel("频数")
                
                # 匹配后
                sns.histplot(
                    data=matched_data, 
                    x='propensity_score', 
                    hue=treatment_col,
                    bins=20,
                    ax=ax2
                )
                ax2.set_title("匹配后倾向得分分布")
                ax2.set_xlabel("倾向得分")
                ax2.set_ylabel("频数")
                
                plt.tight_layout()
                st.pyplot(fig)
                
                # 显示匹配后的数据预览
                st.subheader("匹配后数据预览")
                st.dataframe(matched_data.head())
                
            except Exception as e:
                st.error(f"匹配过程中出错: {e}")
    
    # 如果已经有匹配的数据，显示匹配信息
    elif 'matched_data' in st.session_state:
        st.success(f"数据已匹配! 匹配后样本数: {len(st.session_state['matched_data'])}")
        
        if st.checkbox("显示匹配数据预览"):
            st.subheader("匹配后数据预览")
            st.dataframe(st.session_state['matched_data'].head())
    
    # 添加技术说明
    with st.expander("技术说明"):
        st.markdown("""
        #### 统计匹配的重要性
        
        在观察性研究中，由于缺乏随机分配，处理组和对照组在基线特征上可能存在系统性差异，这会导致选择偏倚和混杂因素的影响。统计匹配方法可以：
        
        1. **减少选择偏倚**：通过平衡处理组和对照组的协变量分布，减少选择偏倚的影响
        2. **控制混杂因素**：通过匹配可能影响处理分配和结果的变量，控制混杂因素
        3. **提高因果推断的有效性**：使观察性研究更接近随机对照试验的设计
        
        #### 倾向得分匹配
        
        倾向得分是样本接受处理的条件概率，通常使用逻辑回归等方法估计：
        
        1. **倾向得分计算**：基于协变量估计每个样本接受处理的概率
        2. **匹配过程**：根据倾向得分的相似性，为处理组样本匹配对照组样本
        3. **平衡评估**：评估匹配前后协变量的平衡性，通常使用标准化均值差异
        
        #### 匹配方法
        
        1. **最近邻匹配**：
           - 为每个处理组样本找到倾向得分最接近的一个或多个对照组样本
           - 可以设置是否允许替换（一个对照组样本是否可以被多次匹配）
        
        2. **半径匹配**：
           - 为每个处理组样本找到倾向得分在指定半径内的所有对照组样本
           - 半径通常设置为倾向得分标准差的一定倍数
        
        #### 平衡性评估
        
        标准化均值差异（Standardized Mean Difference, SMD）是评估协变量平衡性的常用指标：
        
        - SMD = (处理组均值 - 对照组均值) / 合并标准差
        - 通常认为SMD < 0.1表示良好的平衡
        - 匹配后SMD应该比匹配前小，表示平衡性得到改善
        """)

if __name__ == "__main__":
    statistical_matching_page()
