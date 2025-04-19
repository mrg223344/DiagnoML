import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

def tpm_normalization(counts_df):
    """
    将原始计数转换为TPM (Transcripts Per Million)
    
    参数:
    counts_df: 包含基因表达计数的DataFrame
    
    返回:
    DataFrame: TPM标准化后的表达数据
    """
    # 获取基因列（假设非基因列已被排除）
    gene_cols = counts_df.columns
    
    # 计算每个基因的长度（这里我们假设所有基因长度相同，实际应用中应该使用真实的基因长度）
    # 在实际应用中，这里应该是从外部文件导入的基因长度信息
    gene_lengths = pd.Series(1000, index=gene_cols)  # 假设所有基因长度为1000bp
    
    # 第一步：将读数除以基因长度得到RPK (Reads Per Kilobase)
    rpk = counts_df.div(gene_lengths / 1000, axis=1)
    
    # 第二步：计算每个样本的RPK总和，然后除以10^6得到缩放因子
    scaling_factors = rpk.sum(axis=1) / 1e6
    
    # 第三步：将RPK除以缩放因子得到TPM
    tpm = rpk.div(scaling_factors, axis=0)
    
    return tpm

def z_score_normalization(df):
    """
    对表达数据进行z-score标准化
    
    参数:
    df: 包含表达数据的DataFrame
    
    返回:
    DataFrame: z-score标准化后的表达数据
    """
    # 计算每个基因的均值和标准差
    mean = df.mean(axis=0)
    std = df.std(axis=0)
    
    # 进行z-score标准化
    z_scores = df.sub(mean, axis=1).div(std, axis=1)
    
    return z_scores

def log2_transformation(df, pseudo_count=1):
    """
    对表达数据进行log2转换
    
    参数:
    df: 包含表达数据的DataFrame
    pseudo_count: 添加的伪计数，避免log(0)
    
    返回:
    DataFrame: log2转换后的表达数据
    """
    return np.log2(df + pseudo_count)

def plot_expression_distribution(df, title, n_genes=10):
    """
    绘制表达分布图
    
    参数:
    df: 包含表达数据的DataFrame
    title: 图表标题
    n_genes: 要显示的基因数量
    """
    # 选择前n个基因进行可视化
    selected_genes = df.columns[:n_genes]
    data_to_plot = df[selected_genes].melt(var_name='Gene', value_name='Expression')
    
    plt.figure(figsize=(12, 6))
    sns.boxplot(x='Gene', y='Expression', data=data_to_plot)
    plt.title(title)
    plt.xticks(rotation=45)
    plt.tight_layout()
    
    return plt

def expression_correction_page():
    st.title("表达量校正")
    
    st.markdown("""
    ### 表达量校正模块
    
    本模块提供RNA-seq数据的表达量校正功能，包括：
    
    1. **TPM转换**：将原始计数转换为TPM (Transcripts Per Million)
    2. **Log2转换**：对表达数据进行log2转换，使分布更接近正态分布
    3. **Z-score标准化**：对基因表达数据进行z-score标准化
    
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
    
    if len(gene_cols) == 0:
        st.error("未检测到基因表达数据列（以GENE_开头的列）")
        return
    
    # 显示原始数据预览
    st.subheader("原始表达数据预览")
    st.dataframe(data[gene_cols].head())
    
    # 表达量校正选项
    st.subheader("选择校正方法")
    correction_methods = st.multiselect(
        "选择要应用的校正方法",
        ["TPM转换", "Log2转换", "Z-score标准化"],
        default=["TPM转换", "Z-score标准化"]
    )
    
    # 应用选择的校正方法
    expression_data = data[gene_cols].copy()
    corrected_data = data.copy()
    
    if correction_methods:
        with st.spinner("正在进行表达量校正..."):
            # 创建处理前后对比的图表
            col1, col2 = st.columns(2)
            
            with col1:
                st.subheader("校正前表达分布")
                fig_before = plot_expression_distribution(expression_data, "原始表达分布")
                st.pyplot(fig_before)
            
            # 应用校正方法
            if "TPM转换" in correction_methods:
                expression_data = tpm_normalization(expression_data)
                st.success("✅ 已完成TPM转换")
            
            if "Log2转换" in correction_methods:
                expression_data = log2_transformation(expression_data)
                st.success("✅ 已完成Log2转换")
            
            if "Z-score标准化" in correction_methods:
                expression_data = z_score_normalization(expression_data)
                st.success("✅ 已完成Z-score标准化")
            
            # 更新校正后的数据
            for col in gene_cols:
                corrected_data[col] = expression_data[col]
            
            with col2:
                st.subheader("校正后表达分布")
                fig_after = plot_expression_distribution(expression_data, "校正后表达分布")
                st.pyplot(fig_after)
            
            # 显示校正后的数据预览
            st.subheader("校正后表达数据预览")
            st.dataframe(expression_data.head())
            
            # 保存校正后的数据
            if st.button("保存校正后的数据"):
                st.session_state['data'] = corrected_data
                st.session_state['corrected_expression'] = expression_data
                st.success("校正后的数据已保存到会话状态中，可以继续进行后续分析")
    else:
        st.info("请选择至少一种校正方法")
    
    # 添加技术说明
    with st.expander("技术说明"):
        st.markdown("""
        #### TPM (Transcripts Per Million)
        TPM是一种常用的RNA-seq数据标准化方法，它考虑了基因长度和测序深度的影响。计算步骤如下：
        1. 将每个基因的读数除以基因长度（千碱基），得到RPK (Reads Per Kilobase)
        2. 计算每个样本的RPK总和，然后除以10^6得到缩放因子
        3. 将RPK除以缩放因子得到TPM
        
        #### Log2转换
        Log2转换可以使表达数据分布更接近正态分布，减小高表达基因的影响，便于后续分析。为避免log(0)的问题，通常会添加一个小的伪计数。
        
        #### Z-score标准化
        Z-score标准化将每个基因的表达值转换为标准分数，使得每个基因的表达值均值为0，标准差为1。这有助于比较不同基因的表达模式。
        """)

if __name__ == "__main__":
    expression_correction_page()
