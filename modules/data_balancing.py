import streamlit as st
import pandas as pd
import numpy as np
from sklearn.utils import resample
from imblearn.over_sampling import SMOTE
from imblearn.under_sampling import RandomUnderSampler
import matplotlib.pyplot as plt
import seaborn as sns

def data_balancing_page():
    st.title("数据平衡")
    
    st.markdown("""
    ### 数据平衡模块
    
    本模块提供处理不平衡数据集的功能，适用于分类问题中各类别样本数量差异较大的情况。
    
    支持的平衡方法：
    
    1. **欠采样**：从多数类中随机抽取样本，使其数量与少数类相当
    2. **过采样**：对少数类进行随机复制，使其数量与多数类相当
    3. **SMOTE**：合成少数类过采样技术，生成新的少数类样本
    
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
        st.info("使用已拆分的训练集进行数据平衡")
    else:
        # 否则使用全部数据
        data = st.session_state['data']
        st.info("使用全部数据进行数据平衡")
    
    # 分离基因表达数据和临床数据
    # 假设所有以GENE_开头的列是基因表达数据
    gene_cols = [col for col in data.columns if col.startswith('GENE_')]
    clinical_cols = [col for col in data.columns if col not in gene_cols]
    
    # 选择目标变量
    st.subheader("选择目标变量")
    
    # 筛选分类变量
    categorical_cols = [col for col in clinical_cols 
                       if pd.api.types.is_categorical_dtype(data[col]) or 
                       pd.api.types.is_object_dtype(data[col]) or
                       (pd.api.types.is_numeric_dtype(data[col]) and data[col].nunique() < 10)]
    
    if not categorical_cols:
        st.error("未检测到合适的分类变量，数据平衡仅适用于分类问题")
        return
    
    target_col = st.selectbox(
        "选择目标变量",
        categorical_cols,
        index=categorical_cols.index('disease_status') if 'disease_status' in categorical_cols else 0,
        help="选择需要平衡的目标变量，必须是分类变量"
    )
    
    # 显示类别分布
    st.subheader("类别分布")
    
    # 计算类别分布
    class_counts = data[target_col].value_counts()
    class_percentages = data[target_col].value_counts(normalize=True) * 100
    
    # 创建分布数据框
    distribution_df = pd.DataFrame({
        '类别': class_counts.index,
        '样本数': class_counts.values,
        '百分比': class_percentages.values.round(2)
    })
    
    # 显示分布表格
    st.dataframe(distribution_df)
    
    # 绘制分布图
    fig, ax = plt.subplots(figsize=(10, 6))
    sns.barplot(x=class_counts.index.astype(str), y=class_counts.values, ax=ax)
    ax.set_title(f"{target_col} 类别分布")
    ax.set_xlabel("类别")
    ax.set_ylabel("样本数")
    
    # 在柱状图上添加数值标签
    for i, v in enumerate(class_counts.values):
        ax.text(i, v + 0.1, str(v), ha='center')
    
    st.pyplot(fig)
    
    # 判断是否需要平衡
    max_class_count = class_counts.max()
    min_class_count = class_counts.min()
    imbalance_ratio = max_class_count / min_class_count
    
    if imbalance_ratio > 1.5:
        st.warning(f"检测到数据不平衡，最大类与最小类的比例为 {imbalance_ratio:.2f}")
    else:
        st.success(f"数据相对平衡，最大类与最小类的比例为 {imbalance_ratio:.2f}")
    
    # 数据平衡方法
    st.subheader("选择平衡方法")
    
    balance_method = st.selectbox(
        "平衡方法",
        ["欠采样", "过采样", "SMOTE"],
        index=0,
        help="选择处理不平衡数据的方法"
    )
    
    # 设置平衡参数
    if balance_method == "欠采样":
        sampling_strategy = st.selectbox(
            "采样策略",
            ["平衡所有类别", "自定义比例"],
            index=0,
            help="平衡所有类别将使所有类别样本数相等；自定义比例允许设置各类别的相对比例"
        )
        
        if sampling_strategy == "自定义比例":
            st.info("请输入各类别的相对比例，例如：对于二分类问题，0.5表示少数类与多数类的比例为1:2")
            ratio = st.slider("少数类与多数类的比例", 0.1, 1.0, 0.5, 0.1)
        else:
            ratio = 1.0
    
    elif balance_method == "过采样":
        sampling_strategy = st.selectbox(
            "采样策略",
            ["平衡所有类别", "自定义比例"],
            index=0,
            help="平衡所有类别将使所有类别样本数相等；自定义比例允许设置各类别的相对比例"
        )
        
        if sampling_strategy == "自定义比例":
            st.info("请输入各类别的相对比例，例如：对于二分类问题，0.5表示少数类与多数类的比例为1:2")
            ratio = st.slider("少数类与多数类的比例", 0.1, 1.0, 0.5, 0.1)
        else:
            ratio = 1.0
    
    elif balance_method == "SMOTE":
        k_neighbors = st.slider(
            "K近邻数量",
            1, 10, 5,
            help="SMOTE算法中用于生成新样本的近邻数量"
        )
        
        sampling_strategy = "平衡所有类别"
        ratio = 1.0
    
    # 执行数据平衡
    if st.button("执行数据平衡"):
        with st.spinner("正在平衡数据..."):
            # 准备特征和目标变量
            if 'selected_features' in st.session_state:
                # 如果已经进行了特征选择，使用选择的特征
                selected_features = st.session_state['selected_features']
                X = data[selected_features]
                st.info(f"使用特征工程选择的 {len(selected_features)} 个特征")
            else:
                # 否则使用所有基因特征
                X = data[gene_cols]
            
            y = data[target_col]
            
            # 根据选择的方法进行数据平衡
            if balance_method == "欠采样":
                if sampling_strategy == "平衡所有类别":
                    strategy = 'auto'
                else:
                    # 创建自定义比例字典
                    class_labels = y.unique()
                    majority_label = class_counts.idxmax()
                    strategy = {label: int(class_counts[majority_label] * ratio) if label != majority_label else class_counts[majority_label] for label in class_labels}
                
                # 执行欠采样
                undersampler = RandomUnderSampler(sampling_strategy=strategy, random_state=42)
                X_resampled, y_resampled = undersampler.fit_resample(X, y)
                
                st.success(f"欠采样完成! 平衡后样本数: {len(y_resampled)}")
            
            elif balance_method == "过采样":
                if sampling_strategy == "平衡所有类别":
                    strategy = 'auto'
                else:
                    # 创建自定义比例字典
                    class_labels = y.unique()
                    majority_label = class_counts.idxmax()
                    strategy = {label: int(class_counts[majority_label]) if label != majority_label else class_counts[majority_label] for label in class_labels}
                
                # 执行过采样
                oversampler = RandomUnderSampler(sampling_strategy=strategy, random_state=42)
                X_resampled, y_resampled = oversampler.fit_resample(X, y)
                
                st.success(f"过采样完成! 平衡后样本数: {len(y_resampled)}")
            
            elif balance_method == "SMOTE":
                # 执行SMOTE
                try:
                    smote = SMOTE(k_neighbors=k_neighbors, random_state=42)
                    X_resampled, y_resampled = smote.fit_resample(X, y)
                    
                    st.success(f"SMOTE完成! 平衡后样本数: {len(y_resampled)}")
                except ValueError as e:
                    st.error(f"SMOTE错误: {e}")
                    st.info("SMOTE要求少数类样本数量大于k_neighbors参数值，请尝试减小k_neighbors或使用其他平衡方法")
                    return
            
            # 创建平衡后的数据集
            balanced_data = pd.DataFrame(X_resampled, columns=X.columns)
            balanced_data[target_col] = y_resampled
            
            # 添加其他临床特征（如果有）
            other_clinical_cols = [col for col in clinical_cols if col != target_col]
            if other_clinical_cols:
                st.warning("平衡过程中其他临床特征可能会丢失，建议在后续分析中重新添加这些特征")
            
            # 保存到会话状态
            st.session_state['balanced_data'] = balanced_data
            
            # 如果已经拆分了数据，更新训练集
            if 'train_data' in st.session_state:
                st.session_state['train_data'] = balanced_data
                st.info("已更新训练集为平衡后的数据")
            
            # 显示平衡后的类别分布
            st.subheader("平衡后的类别分布")
            
            # 计算平衡后的类别分布
            balanced_class_counts = balanced_data[target_col].value_counts()
            balanced_class_percentages = balanced_data[target_col].value_counts(normalize=True) * 100
            
            # 创建分布数据框
            balanced_distribution_df = pd.DataFrame({
                '类别': balanced_class_counts.index,
                '样本数': balanced_class_counts.values,
                '百分比': balanced_class_percentages.values.round(2)
            })
            
            # 显示分布表格
            st.dataframe(balanced_distribution_df)
            
            # 绘制平衡前后的对比图
            fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))
            
            # 平衡前
            sns.barplot(x=class_counts.index.astype(str), y=class_counts.values, ax=ax1)
            ax1.set_title("平衡前类别分布")
            ax1.set_xlabel("类别")
            ax1.set_ylabel("样本数")
            
            # 平衡后
            sns.barplot(x=balanced_class_counts.index.astype(str), y=balanced_class_counts.values, ax=ax2)
            ax2.set_title("平衡后类别分布")
            ax2.set_xlabel("类别")
            ax2.set_ylabel("样本数")
            
            plt.tight_layout()
            st.pyplot(fig)
            
            # 显示平衡后的数据预览
            st.subheader("平衡后数据预览")
            st.dataframe(balanced_data.head())
    
    # 如果已经有平衡的数据，显示平衡信息
    elif 'balanced_data' in st.session_state:
        st.success(f"数据已平衡! 平衡后样本数: {len(st.session_state['balanced_data'])}")
        
        if st.checkbox("显示平衡数据预览"):
            st.subheader("平衡后数据预览")
            st.dataframe(st.session_state['balanced_data'].head())
    
    # 添加技术说明
    with st.expander("技术说明"):
        st.markdown("""
        #### 数据不平衡问题
        
        在分类问题中，当不同类别的样本数量差异较大时，就会出现数据不平衡问题。这可能导致：
        
        1. **模型偏向多数类**：模型倾向于预测多数类，忽略少数类
        2. **评估指标失真**：准确率等指标可能掩盖模型在少数类上的表现不佳
        3. **少数类特征学习不足**：模型难以学习少数类的特征模式
        
        #### 处理不平衡数据的方法
        
        1. **欠采样**：
           - 从多数类中随机抽取样本，使其数量与少数类相当
           - 优点：减少训练时间，避免过拟合
           - 缺点：可能丢失多数类中的重要信息
        
        2. **过采样**：
           - 对少数类进行随机复制，使其数量与多数类相当
           - 优点：不丢失信息
           - 缺点：可能导致过拟合，因为少数类样本被重复使用
        
        3. **SMOTE (Synthetic Minority Over-sampling Technique)**：
           - 通过在少数类样本之间插值生成新的合成样本
           - 优点：生成新样本而不是简单复制，减轻过拟合
           - 缺点：可能在类别边界处生成不合适的样本
        
        #### 选择合适的平衡方法
        
        - 当数据集较大且多数类样本信息冗余时，可以考虑欠采样
        - 当数据集较小且不能承受信息损失时，可以考虑过采样或SMOTE
        - SMOTE通常是一个较好的折中方案，但需要调整k_neighbors参数
        """)

if __name__ == "__main__":
    data_balancing_page()
