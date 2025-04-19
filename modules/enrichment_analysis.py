import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import plotly.express as px
import requests
import json
from io import StringIO

def enrichment_analysis_page():
    st.title("富集分析")
    
    st.markdown("""
    ### 富集分析模块
    
    本模块提供基因功能富集分析功能，帮助您理解基因集的生物学功能和通路。
    
    支持的富集分析类型：
    
    1. **GO富集分析**：基因本体论（Gene Ontology）富集分析
    2. **KEGG通路富集分析**：京都基因与基因组百科全书（KEGG）通路富集分析
    3. **基因集富集分析(GSEA)**：评估基因集在排序列表中的富集程度
    
    请先训练机器学习模型或选择感兴趣的基因集，然后使用本模块进行富集分析。
    """)
    
    # 检查是否有数据加载
    if 'data' not in st.session_state:
        st.warning("请先在侧边栏上传数据或生成示例数据")
        return
    
    # 获取数据
    data = st.session_state['data']
    
    # 分离基因表达数据和临床数据
    gene_cols = [col for col in data.columns if col.startswith('GENE_')]
    clinical_cols = [col for col in data.columns if col not in gene_cols]
    
    if len(gene_cols) == 0:
        st.error("未检测到基因表达数据列（以GENE_开头的列）")
        return
    
    # 选择基因集来源
    st.subheader("选择基因集来源")
    
    gene_source = st.radio(
        "基因集来源",
        ["模型特征重要性", "差异表达分析", "手动输入基因列表"],
        index=0
    )
    
    selected_genes = []
    
    if gene_source == "模型特征重要性":
        if 'model' in st.session_state and 'model_features' in st.session_state:
            # 使用模型选择的特征
            model_features = st.session_state['model_features']
            
            # 显示模型特征
            st.write(f"模型使用的特征数量: {len(model_features)}")
            
            # 选择要使用的特征数量
            n_top_features = st.slider(
                "选择要用于富集分析的顶部特征数量", 
                10, min(100, len(model_features)), 
                min(30, len(model_features))
            )
            
            # 获取顶部特征
            selected_genes = model_features[:n_top_features]
            
            # 显示选择的特征
            st.write("选择的基因特征:")
            st.write(", ".join(selected_genes))
        else:
            st.warning("未找到训练好的模型，请先在机器学习模型模块训练模型")
    
    elif gene_source == "差异表达分析":
        # 选择分组变量
        group_var = st.selectbox(
            "选择分组变量",
            clinical_cols,
            index=clinical_cols.index('disease_status') if 'disease_status' in clinical_cols else 0
        )
        
        # 检查分组变量是否为分类变量
        if pd.api.types.is_categorical_dtype(data[group_var]) or data[group_var].nunique() < 10:
            # 获取分组
            groups = data[group_var].unique()
            
            if len(groups) == 2:
                # 二分类比较
                group1 = groups[0]
                group2 = groups[1]
                
                st.write(f"比较组别: {group1} vs {group2}")
                
                # 执行差异表达分析
                if st.button("执行差异表达分析"):
                    with st.spinner("正在执行差异表达分析..."):
                        # 准备数据
                        group1_data = data[data[group_var] == group1][gene_cols]
                        group2_data = data[data[group_var] == group2][gene_cols]
                        
                        # 计算每个基因的平均表达量
                        group1_mean = group1_data.mean()
                        group2_mean = group2_data.mean()
                        
                        # 计算差异倍数（fold change）
                        fold_change = group2_mean / group1_mean
                        log2_fold_change = np.log2(fold_change)
                        
                        # 执行t检验
                        from scipy.stats import ttest_ind
                        
                        p_values = []
                        for gene in gene_cols:
                            t_stat, p_val = ttest_ind(
                                group1_data[gene].dropna(),
                                group2_data[gene].dropna(),
                                equal_var=False  # 不假设等方差
                            )
                            p_values.append(p_val)
                        
                        # 多重检验校正
                        from statsmodels.stats.multitest import multipletests
                        
                        _, p_adjusted, _, _ = multipletests(p_values, method='fdr_bh')
                        
                        # 创建结果数据框
                        results = pd.DataFrame({
                            'Gene': gene_cols,
                            'log2FoldChange': log2_fold_change,
                            'pvalue': p_values,
                            'padj': p_adjusted
                        })
                        
                        # 按校正后p值排序
                        results = results.sort_values('padj')
                        
                        # 显示结果
                        st.subheader("差异表达分析结果")
                        st.dataframe(results.round(4))
                        
                        # 绘制火山图
                        fig, ax = plt.subplots(figsize=(10, 8))
                        
                        # 设置显著性阈值
                        padj_threshold = 0.05
                        lfc_threshold = 1  # log2 fold change > 1 表示倍数变化 > 2
                        
                        # 标记显著差异的基因
                        results['Significant'] = 'Not Significant'
                        results.loc[(results['padj'] < padj_threshold) & (results['log2FoldChange'] > lfc_threshold), 'Significant'] = 'Up-regulated'
                        results.loc[(results['padj'] < padj_threshold) & (results['log2FoldChange'] < -lfc_threshold), 'Significant'] = 'Down-regulated'
                        
                        # 绘制散点图
                        sns.scatterplot(
                            data=results,
                            x='log2FoldChange',
                            y=-np.log10(results['padj']),
                            hue='Significant',
                            palette={'Up-regulated': 'red', 'Down-regulated': 'blue', 'Not Significant': 'gray'},
                            alpha=0.7,
                            ax=ax
                        )
                        
                        # 添加阈值线
                        ax.axhline(-np.log10(padj_threshold), linestyle='--', color='gray')
                        ax.axvline(lfc_threshold, linestyle='--', color='gray')
                        ax.axvline(-lfc_threshold, linestyle='--', color='gray')
                        
                        ax.set_xlabel('log2 Fold Change')
                        ax.set_ylabel('-log10(adjusted p-value)')
                        ax.set_title('Volcano Plot')
                        
                        st.pyplot(fig)
                        
                        # 获取显著上调和下调的基因
                        up_genes = results[results['Significant'] == 'Up-regulated']['Gene'].tolist()
                        down_genes = results[results['Significant'] == 'Down-regulated']['Gene'].tolist()
                        
                        st.write(f"显著上调基因数量: {len(up_genes)}")
                        st.write(f"显著下调基因数量: {len(down_genes)}")
                        
                        # 选择用于富集分析的基因集
                        gene_set_choice = st.radio(
                            "选择用于富集分析的基因集",
                            ["上调基因", "下调基因", "所有显著差异基因"],
                            index=0
                        )
                        
                        if gene_set_choice == "上调基因":
                            selected_genes = up_genes
                        elif gene_set_choice == "下调基因":
                            selected_genes = down_genes
                        else:
                            selected_genes = up_genes + down_genes
                        
                        # 显示选择的基因
                        if selected_genes:
                            st.write("选择的基因:")
                            st.write(", ".join(selected_genes))
                        else:
                            st.warning("未找到显著差异的基因")
            else:
                st.warning("差异表达分析当前仅支持两组比较")
        else:
            st.warning("请选择分类变量作为分组变量")
    
    elif gene_source == "手动输入基因列表":
        # 手动输入基因列表
        gene_list_input = st.text_area(
            "输入基因列表（每行一个基因名称）",
            height=150,
            help="输入感兴趣的基因列表，平台将自动匹配数据集中的对应基因"
        )
        
        if gene_list_input:
            # 解析输入的基因列表
            input_genes = [gene.strip() for gene in gene_list_input.split('\n') if gene.strip()]
            
            # 匹配数据集中的基因
            matched_genes = []
            for gene in input_genes:
                matched = [col for col in gene_cols if gene in col]
                matched_genes.extend(matched)
            
            if matched_genes:
                selected_genes = matched_genes
                st.success(f"成功匹配 {len(selected_genes)} 个基因")
                
                # 显示匹配的基因
                st.write("匹配的基因:")
                st.write(", ".join(selected_genes))
            else:
                st.error("未能匹配任何基因，请检查输入的基因名称")
    
    # 执行富集分析
    if selected_genes:
        st.subheader("富集分析")
        
        # 选择富集分析类型
        enrichment_type = st.radio(
            "选择富集分析类型",
            ["GO富集分析", "KEGG通路富集分析", "基因集富集分析(GSEA)"],
            index=0
        )
        
        # 提取基因名称（去除GENE_前缀）
        gene_names = [gene.replace('GENE_', '') for gene in selected_genes]
        
        # 模拟富集分析（实际应用中应使用真实的API调用）
        if st.button("执行富集分析"):
            with st.spinner("正在执行富集分析..."):
                # 模拟富集分析结果
                if enrichment_type == "GO富集分析":
                    # 模拟GO富集分析结果
                    go_terms = [
                        "GO:0006915 - apoptotic process",
                        "GO:0007165 - signal transduction",
                        "GO:0006355 - regulation of transcription, DNA-templated",
                        "GO:0006954 - inflammatory response",
                        "GO:0007275 - multicellular organism development",
                        "GO:0045087 - innate immune response",
                        "GO:0006468 - protein phosphorylation",
                        "GO:0016032 - viral process",
                        "GO:0006955 - immune response",
                        "GO:0007155 - cell adhesion"
                    ]
                    
                    # 生成随机p值和富集比
                    np.random.seed(42)
                    p_values = np.random.uniform(0.0001, 0.05, len(go_terms))
                    enrichment_ratios = np.random.uniform(1.5, 5.0, len(go_terms))
                    gene_counts = np.random.randint(3, len(gene_names), len(go_terms))
                    
                    # 创建结果数据框
                    results = pd.DataFrame({
                        'Term': go_terms,
                        'Count': gene_counts,
                        'EnrichmentRatio': enrichment_ratios,
                        'PValue': p_values,
                        'AdjustedPValue': p_values * 1.2  # 简单模拟校正后的p值
                    })
                    
                    # 按p值排序
                    results = results.sort_values('PValue')
                    
                    # 显示结果
                    st.subheader("GO富集分析结果")
                    st.dataframe(results.round(4))
                    
                    # 绘制富集分析条形图
                    fig, ax = plt.subplots(figsize=(10, 8))
                    
                    # 选择前10个结果
                    plot_data = results.head(10).copy()
                    
                    # 绘制条形图
                    bars = sns.barplot(
                        data=plot_data,
                        y='Term',
                        x='EnrichmentRatio',
                        ax=ax,
                        palette='viridis'
                    )
                    
                    # 添加基因计数标签
                    for i, count in enumerate(plot_data['Count']):
                        ax.text(
                            plot_data['EnrichmentRatio'].iloc[i] + 0.1,
                            i,
                            f"n={count}",
                            va='center'
                        )
                    
                    ax.set_xlabel('Enrichment Ratio')
                    ax.set_ylabel('GO Term')
                    ax.set_title('GO Enrichment Analysis')
                    
                    st.pyplot(fig)
                    
                    # 绘制气泡图
                    fig, ax = plt.subplots(figsize=(12, 8))
                    
                    # 创建气泡图
                    scatter = ax.scatter(
                        plot_data['EnrichmentRatio'],
                        range(len(plot_data)),
                        s=plot_data['Count'] * 20,  # 气泡大小基于基因计数
                        c=-np.log10(plot_data['PValue']),  # 颜色基于p值
                        cmap='viridis',
                        alpha=0.7
                    )
                    
                    # 添加颜色条
                    cbar = plt.colorbar(scatter)
                    cbar.set_label('-log10(p-value)')
                    
                    # 设置y轴标签
                    ax.set_yticks(range(len(plot_data)))
                    ax.set_yticklabels(plot_data['Term'])
                    
                    ax.set_xlabel('Enrichment Ratio')
                    ax.set_title('GO Enrichment Analysis')
                    
                    st.pyplot(fig)
                
                elif enrichment_type == "KEGG通路富集分析":
                    # 模拟KEGG通路富集分析结果
                    kegg_pathways = [
                        "hsa04010 - MAPK signaling pathway",
                        "hsa04151 - PI3K-Akt signaling pathway",
                        "hsa04060 - Cytokine-cytokine receptor interaction",
                        "hsa04620 - Toll-like receptor signaling pathway",
                        "hsa04630 - JAK-STAT signaling pathway",
                        "hsa04210 - Apoptosis",
                        "hsa04668 - TNF signaling pathway",
                        "hsa04066 - HIF-1 signaling pathway",
                        "hsa04310 - Wnt signaling pathway",
                        "hsa04350 - TGF-beta signaling pathway"
                    ]
                    
                    # 生成随机p值和富集比
                    np.random.seed(42)
                    p_values = np.random.uniform(0.0001, 0.05, len(kegg_pathways))
                    enrichment_ratios = np.random.uniform(1.5, 5.0, len(kegg_pathways))
                    gene_counts = np.random.randint(3, len(gene_names), len(kegg_pathways))
                    
                    # 创建结果数据框
                    results = pd.DataFrame({
                        'Pathway': kegg_pathways,
                        'Count': gene_counts,
                        'EnrichmentRatio': enrichment_ratios,
                        'PValue': p_values,
                        'AdjustedPValue': p_values * 1.2  # 简单模拟校正后的p值
                    })
                    
                    # 按p值排序
                    results = results.sort_values('PValue')
                    
                    # 显示结果
                    st.subheader("KEGG通路富集分析结果")
                    st.dataframe(results.round(4))
                    
                    # 绘制富集分析条形图
                    fig, ax = plt.subplots(figsize=(10, 8))
                    
                    # 选择前10个结果
                    plot_data = results.head(10).copy()
                    
                    # 绘制条形图
                    bars = sns.barplot(
                        data=plot_data,
                        y='Pathway',
                        x='EnrichmentRatio',
                        ax=ax,
                        palette='viridis'
                    )
                    
                    # 添加基因计数标签
                    for i, count in enumerate(plot_data['Count']):
                        ax.text(
                            plot_data['EnrichmentRatio'].iloc[i] + 0.1,
                            i,
                            f"n={count}",
                            va='center'
                        )
                    
                    ax.set_xlabel('Enrichment Ratio')
                    ax.set_ylabel('KEGG Pathway')
                    ax.set_title('KEGG Pathway Enrichment Analysis')
                    
                    st.pyplot(fig)
                    
                    # 绘制网络图
                    st.subheader("通路-基因网络图")
                    
                    # 使用Plotly创建网络图
                    # 为简化起见，这里只创建一个简单的示例网络
                    
                    # 创建节点
                    nodes = []
                    # 添加通路节点
                    for i, pathway in enumerate(plot_data['Pathway']):
                        nodes.append({
                            'id': f"pathway_{i}",
                            'label': pathway,
                            'type': 'pathway'
                        })
                    
                    # 添加基因节点
                    for i, gene in enumerate(gene_names[:15]):  # 限制基因数量
                        nodes.append({
                            'id': f"gene_{i}",
                            'label': gene,
                            'type': 'gene'
                        })
                    
                    # 创建边
                    edges = []
                    # 随机连接通路和基因
                    np.random.seed(42)
                    for i in range(len(plot_data)):
                        # 每个通路连接到几个随机基因
                        n_connections = min(plot_data['Count'].iloc[i], 5)  # 限制连接数
                        gene_indices = np.random.choice(min(15, len(gene_names)), n_connections, replace=False)
                        
                        for gene_idx in gene_indices:
                            edges.append({
                                'source': f"pathway_{i}",
                                'target': f"gene_{gene_idx}"
                            })
                    
                    # 使用文本描述替代网络图
                    st.info("通路-基因网络图展示了富集通路与基因之间的关系。在实际应用中，可以使用NetworkX或Cytoscape.js创建交互式网络图。")
                    
                    # 显示通路-基因关联表
                    st.subheader("通路-基因关联")
                    
                    # 创建关联表
                    associations = []
                    for edge in edges:
                        pathway_id = int(edge['source'].split('_')[1])
                        gene_id = int(edge['target'].split('_')[1])
                        
                        associations.append({
                            'Pathway': plot_data['Pathway'].iloc[pathway_id],
                            'Gene': gene_names[gene_id]
                        })
                    
                    associations_df = pd.DataFrame(associations)
                    st.dataframe(associations_df)
                
                elif enrichment_type == "基因集富集分析(GSEA)":
                    # 模拟GSEA结果
                    gene_sets = [
                        "HALLMARK_INFLAMMATORY_RESPONSE",
                        "HALLMARK_INTERFERON_GAMMA_RESPONSE",
                        "HALLMARK_INTERFERON_ALPHA_RESPONSE",
                        "HALLMARK_TNFA_SIGNALING_VIA_NFKB",
                        "HALLMARK_IL6_JAK_STAT3_SIGNALING",
                        "HALLMARK_COMPLEMENT",
                        "HALLMARK_APOPTOSIS",
                        "HALLMARK_P53_PATHWAY",
                        "HALLMARK_HYPOXIA",
                        "HALLMARK_EPITHELIAL_MESENCHYMAL_TRANSITION"
                    ]
                    
                    # 生成随机富集分数和p值
                    np.random.seed(42)
                    enrichment_scores = np.random.uniform(-0.8, 0.8, len(gene_sets))
                    p_values = np.random.uniform(0.0001, 0.05, len(gene_sets))
                    fdr_values = p_values * 1.2  # 简单模拟FDR
                    
                    # 创建结果数据框
                    results = pd.DataFrame({
                        'GeneSet': gene_sets,
                        'EnrichmentScore': enrichment_scores,
                        'NormalizedEnrichmentScore': enrichment_scores * 1.2,  # 简单模拟NES
                        'PValue': p_values,
                        'FDR': fdr_values
                    })
                    
                    # 按富集分数绝对值排序
                    results = results.sort_values('EnrichmentScore', key=abs, ascending=False)
                    
                    # 显示结果
                    st.subheader("GSEA结果")
                    st.dataframe(results.round(4))
                    
                    # 绘制富集分数条形图
                    fig, ax = plt.subplots(figsize=(10, 8))
                    
                    # 选择前10个结果
                    plot_data = results.head(10).copy()
                    
                    # 设置条形图颜色
                    colors = ['red' if score > 0 else 'blue' for score in plot_data['EnrichmentScore']]
                    
                    # 绘制条形图
                    bars = ax.barh(
                        plot_data['GeneSet'],
                        plot_data['EnrichmentScore'],
                        color=colors
                    )
                    
                    ax.axvline(x=0, color='black', linestyle='-', linewidth=0.5)
                    ax.set_xlabel('Enrichment Score')
                    ax.set_ylabel('Gene Set')
                    ax.set_title('Gene Set Enrichment Analysis')
                    
                    st.pyplot(fig)
                    
                    # 绘制富集图
                    st.subheader("富集图示例")
                    
                    # 模拟富集图数据
                    # 在实际应用中，这应该是从GSEA结果中提取的
                    
                    # 选择一个基因集进行展示
                    selected_gene_set = plot_data['GeneSet'].iloc[0]
                    
                    # 模拟运行富集分数
                    n_genes = 100
                    running_score = np.cumsum(np.random.normal(0, 0.1, n_genes))
                    running_score = running_score / np.max(np.abs(running_score)) * 0.6
                    
                    # 模拟命中位置
                    hit_indices = np.sort(np.random.choice(n_genes, 20, replace=False))
                    
                    # 创建富集图
                    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(10, 8), gridspec_kw={'height_ratios': [3, 1]})
                    
                    # 绘制运行富集分数
                    ax1.plot(range(n_genes), running_score, color='green')
                    ax1.axhline(y=0, color='black', linestyle='-', linewidth=0.5)
                    
                    # 标记最大富集分数位置
                    max_es_idx = np.argmax(np.abs(running_score))
                    ax1.scatter(max_es_idx, running_score[max_es_idx], color='red', s=50)
                    
                    ax1.set_ylabel('Enrichment Score')
                    ax1.set_title(f'Enrichment Plot: {selected_gene_set}')
                    
                    # 绘制命中位置
                    ax2.vlines(hit_indices, 0, 1, color='black')
                    ax2.set_ylim([0, 1])
                    ax2.set_xlabel('Rank in Ordered Dataset')
                    ax2.set_ylabel('Hits')
                    ax2.set_yticks([])
                    
                    plt.tight_layout()
                    st.pyplot(fig)
    
    # 添加技术说明
    with st.expander("技术说明"):
        st.markdown("""
        #### 富集分析的重要性
        
        富集分析是解释基因列表生物学意义的重要工具：
        
        1. **揭示功能模块**：识别基因集中过度表示的生物学功能或通路
        2. **解释分子机制**：帮助理解疾病或表型的潜在分子机制
        3. **发现新靶点**：识别潜在的治疗靶点或生物标志物
        4. **验证实验结果**：确认实验结果与已知生物学知识的一致性
        
        #### 富集分析方法
        
        1. **GO富集分析**：
           - 基于基因本体论（Gene Ontology）数据库
           - 包括生物学过程（BP）、分子功能（MF）和细胞组分（CC）三个方面
           - 使用超几何检验评估富集显著性
        
        2. **KEGG通路富集分析**：
           - 基于京都基因与基因组百科全书（KEGG）数据库
           - 关注代谢通路、信号转导和疾病相关通路
           - 提供通路图可视化
        
        3. **基因集富集分析(GSEA)**：
           - 考虑整个基因表达谱，而不仅仅是差异表达基因
           - 评估预定义基因集在排序基因列表中的富集程度
           - 适用于检测微小但协调一致的表达变化
        
        #### 结果解读
        
        - **富集比（Enrichment Ratio）**：观察到的基因数与期望基因数的比值，越高表示富集程度越高
        - **P值**：富集的统计显著性，通常使用多重检验校正（如FDR）
        - **基因计数**：富集在特定功能或通路中的基因数量
        - **富集分数（GSEA）**：反映基因集在排序列表中的富集程度和方向
        """)

if __name__ == "__main__":
    enrichment_analysis_page()
