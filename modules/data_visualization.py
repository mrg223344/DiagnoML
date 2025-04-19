import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
import time

def data_visualization_page():
    st.title("数据可视化")
    
    st.markdown("""
    ### 数据可视化模块
    
    本模块提供多种可视化方法，帮助您探索和理解RNA-seq数据的特征和模式。
    
    支持的可视化类型：
    
    1. **基本统计可视化**：分类变量分布、数值变量分布等
    2. **相关性分析**：热图、散点图矩阵等
    3. **降维可视化**：PCA、t-SNE等
    4. **表达模式可视化**：热图、箱线图等
    5. **交互式可视化**：使用Plotly创建交互式图表
    
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
    
    # 显示数据基本信息
    st.subheader("数据基本信息")
    st.write(f"样本数量: {data.shape[0]}")
    st.write(f"基因特征数量: {len(gene_cols)}")
    st.write(f"临床特征数量: {len(clinical_cols)}")
    
    # 可视化类型选择
    st.subheader("选择可视化类型")
    
    viz_type = st.selectbox(
        "可视化类型",
        ["基本统计可视化", "相关性分析", "降维可视化", "表达模式可视化", "交互式可视化"],
        index=0
    )
    
    # 基本统计可视化
    if viz_type == "基本统计可视化":
        st.markdown("### 基本统计可视化")
        
        # 选择可视化子类型
        basic_viz_type = st.radio(
            "选择基本统计可视化类型",
            ["分类变量分布", "数值变量分布", "基因表达分布"],
            index=0
        )
        
        if basic_viz_type == "分类变量分布":
            # 筛选分类变量
            categorical_cols = [col for col in clinical_cols 
                               if pd.api.types.is_categorical_dtype(data[col]) or 
                               pd.api.types.is_object_dtype(data[col]) or
                               (pd.api.types.is_numeric_dtype(data[col]) and data[col].nunique() < 10)]
            
            if not categorical_cols:
                st.warning("未检测到分类变量")
                return
            
            # 选择要可视化的分类变量
            selected_cat_cols = st.multiselect(
                "选择要可视化的分类变量",
                categorical_cols,
                default=categorical_cols[:min(3, len(categorical_cols))]
            )
            
            if selected_cat_cols:
                # 选择可视化方法
                viz_method = st.radio(
                    "选择可视化方法",
                    ["条形图", "饼图", "计数图"],
                    index=0
                )
                
                # 创建可视化
                for col in selected_cat_cols:
                    st.write(f"**{col}** 的分布")
                    
                    # 计算分布
                    value_counts = data[col].value_counts()
                    
                    if viz_method == "条形图":
                        fig, ax = plt.subplots(figsize=(10, 6))
                        sns.barplot(x=value_counts.index.astype(str), y=value_counts.values, ax=ax)
                        ax.set_title(f"{col} 分布")
                        ax.set_xlabel(col)
                        ax.set_ylabel("计数")
                        plt.xticks(rotation=45)
                        st.pyplot(fig)
                    
                    elif viz_method == "饼图":
                        fig, ax = plt.subplots(figsize=(10, 6))
                        ax.pie(value_counts.values, labels=value_counts.index.astype(str), autopct='%1.1f%%')
                        ax.set_title(f"{col} 分布")
                        st.pyplot(fig)
                    
                    elif viz_method == "计数图":
                        fig, ax = plt.subplots(figsize=(10, 6))
                        sns.countplot(data=data, x=col, ax=ax)
                        ax.set_title(f"{col} 分布")
                        ax.set_xlabel(col)
                        ax.set_ylabel("计数")
                        plt.xticks(rotation=45)
                        st.pyplot(fig)
        
        elif basic_viz_type == "数值变量分布":
            # 筛选数值变量
            numeric_cols = [col for col in clinical_cols 
                           if pd.api.types.is_numeric_dtype(data[col]) and 
                           data[col].nunique() > 10]
            
            if not numeric_cols:
                st.warning("未检测到数值变量")
                return
            
            # 选择要可视化的数值变量
            selected_num_cols = st.multiselect(
                "选择要可视化的数值变量",
                numeric_cols,
                default=numeric_cols[:min(3, len(numeric_cols))]
            )
            
            if selected_num_cols:
                # 选择可视化方法
                viz_method = st.radio(
                    "选择可视化方法",
                    ["直方图", "箱线图", "小提琴图", "核密度图"],
                    index=0
                )
                
                # 创建可视化
                for col in selected_num_cols:
                    st.write(f"**{col}** 的分布")
                    
                    if viz_method == "直方图":
                        fig, ax = plt.subplots(figsize=(10, 6))
                        sns.histplot(data=data, x=col, kde=True, ax=ax)
                        ax.set_title(f"{col} 分布")
                        ax.set_xlabel(col)
                        ax.set_ylabel("频数")
                        st.pyplot(fig)
                    
                    elif viz_method == "箱线图":
                        fig, ax = plt.subplots(figsize=(10, 6))
                        sns.boxplot(data=data, x=col, ax=ax)
                        ax.set_title(f"{col} 分布")
                        ax.set_xlabel(col)
                        st.pyplot(fig)
                    
                    elif viz_method == "小提琴图":
                        fig, ax = plt.subplots(figsize=(10, 6))
                        sns.violinplot(data=data, x=col, ax=ax)
                        ax.set_title(f"{col} 分布")
                        ax.set_xlabel(col)
                        st.pyplot(fig)
                    
                    elif viz_method == "核密度图":
                        fig, ax = plt.subplots(figsize=(10, 6))
                        sns.kdeplot(data=data[col], ax=ax, fill=True)
                        ax.set_title(f"{col} 分布")
                        ax.set_xlabel(col)
                        ax.set_ylabel("密度")
                        st.pyplot(fig)
        
        elif basic_viz_type == "基因表达分布":
            # 选择要可视化的基因
            n_genes = st.slider("选择要可视化的基因数量", 5, 20, 10)
            
            # 随机选择基因
            selected_genes = st.multiselect(
                "选择要可视化的基因",
                gene_cols,
                default=gene_cols[:min(n_genes, len(gene_cols))]
            )
            
            if selected_genes:
                # 选择可视化方法
                viz_method = st.radio(
                    "选择可视化方法",
                    ["箱线图", "小提琴图", "条形图", "热图"],
                    index=0
                )
                
                if viz_method in ["箱线图", "小提琴图", "条形图"]:
                    # 准备数据
                    gene_data = data[selected_genes].melt(var_name='Gene', value_name='Expression')
                    
                    if viz_method == "箱线图":
                        fig, ax = plt.subplots(figsize=(12, 6))
                        sns.boxplot(data=gene_data, x='Gene', y='Expression', ax=ax)
                        ax.set_title("基因表达分布")
                        ax.set_xlabel("基因")
                        ax.set_ylabel("表达量")
                        plt.xticks(rotation=45)
                        st.pyplot(fig)
                    
                    elif viz_method == "小提琴图":
                        fig, ax = plt.subplots(figsize=(12, 6))
                        sns.violinplot(data=gene_data, x='Gene', y='Expression', ax=ax)
                        ax.set_title("基因表达分布")
                        ax.set_xlabel("基因")
                        ax.set_ylabel("表达量")
                        plt.xticks(rotation=45)
                        st.pyplot(fig)
                    
                    elif viz_method == "条形图":
                        # 计算每个基因的平均表达量
                        gene_means = data[selected_genes].mean()
                        
                        fig, ax = plt.subplots(figsize=(12, 6))
                        sns.barplot(x=gene_means.index, y=gene_means.values, ax=ax)
                        ax.set_title("基因平均表达量")
                        ax.set_xlabel("基因")
                        ax.set_ylabel("平均表达量")
                        plt.xticks(rotation=45)
                        st.pyplot(fig)
                
                elif viz_method == "热图":
                    # 准备数据
                    gene_data = data[selected_genes]
                    
                    fig, ax = plt.subplots(figsize=(12, 8))
                    sns.heatmap(gene_data.T, cmap="viridis", ax=ax)
                    ax.set_title("基因表达热图")
                    ax.set_xlabel("样本")
                    ax.set_ylabel("基因")
                    st.pyplot(fig)
    
    # 相关性分析
    elif viz_type == "相关性分析":
        st.markdown("### 相关性分析")
        
        # 选择相关性分析类型
        corr_type = st.radio(
            "选择相关性分析类型",
            ["基因间相关性", "基因与临床特征相关性", "散点图矩阵"],
            index=0
        )
        
        if corr_type == "基因间相关性":
            # 选择要分析的基因
            n_genes = st.slider("选择要分析的基因数量", 5, 20, 10)
            
            # 随机选择基因
            selected_genes = st.multiselect(
                "选择要分析的基因",
                gene_cols,
                default=gene_cols[:min(n_genes, len(gene_cols))]
            )
            
            if selected_genes:
                # 计算相关性矩阵
                corr_matrix = data[selected_genes].corr()
                
                # 绘制热图
                fig, ax = plt.subplots(figsize=(12, 10))
                sns.heatmap(corr_matrix, annot=True, cmap="coolwarm", vmin=-1, vmax=1, ax=ax)
                ax.set_title("基因间相关性热图")
                st.pyplot(fig)
        
        elif corr_type == "基因与临床特征相关性":
            # 筛选数值型临床特征
            numeric_clinical_cols = [col for col in clinical_cols 
                                    if pd.api.types.is_numeric_dtype(data[col]) and 
                                    data[col].nunique() > 10]
            
            if not numeric_clinical_cols:
                st.warning("未检测到数值型临床特征")
                return
            
            # 选择要分析的临床特征
            selected_clinical = st.multiselect(
                "选择要分析的临床特征",
                numeric_clinical_cols,
                default=numeric_clinical_cols[:min(3, len(numeric_clinical_cols))]
            )
            
            # 选择要分析的基因
            n_genes = st.slider("选择要分析的基因数量", 5, 20, 10)
            
            selected_genes = st.multiselect(
                "选择要分析的基因",
                gene_cols,
                default=gene_cols[:min(n_genes, len(gene_cols))]
            )
            
            if selected_clinical and selected_genes:
                # 计算相关性矩阵
                corr_matrix = data[selected_genes + selected_clinical].corr()
                
                # 提取基因与临床特征的相关性
                gene_clinical_corr = corr_matrix.loc[selected_genes, selected_clinical]
                
                # 绘制热图
                fig, ax = plt.subplots(figsize=(12, 10))
                sns.heatmap(gene_clinical_corr, annot=True, cmap="coolwarm", vmin=-1, vmax=1, ax=ax)
                ax.set_title("基因与临床特征相关性热图")
                st.pyplot(fig)
        
        elif corr_type == "散点图矩阵":
            # 选择要分析的变量
            n_vars = st.slider("选择要分析的变量数量", 3, 6, 4)
            
            # 筛选数值型变量
            numeric_cols = [col for col in data.columns 
                           if pd.api.types.is_numeric_dtype(data[col]) and 
                           data[col].nunique() > 10]
            
            selected_vars = st.multiselect(
                "选择要分析的变量",
                numeric_cols,
                default=numeric_cols[:min(n_vars, len(numeric_cols))]
            )
            
            if selected_vars:
                # 绘制散点图矩阵
                fig = sns.pairplot(data[selected_vars], diag_kind="kde")
                fig.fig.suptitle("变量间散点图矩阵", y=1.02)
                st.pyplot(fig)
    
    # 降维可视化
    elif viz_type == "降维可视化":
        st.markdown("### 降维可视化")
        
        # 选择降维方法
        dim_reduction_method = st.radio(
            "选择降维方法",
            ["PCA", "t-SNE"],
            index=0
        )
        
        # 选择颜色标记变量
        color_var = st.selectbox(
            "选择颜色标记变量",
            clinical_cols,
            index=clinical_cols.index('disease_status') if 'disease_status' in clinical_cols else 0
        )
        
        # 执行降维
        if st.button("执行降维可视化"):
            with st.spinner("正在执行降维..."):
                # 准备数据
                X = data[gene_cols]
                
                # 标准化数据
                from sklearn.preprocessing import StandardScaler
                X_scaled = StandardScaler().fit_transform(X)
                
                if dim_reduction_method == "PCA":
                    # 执行PCA
                    pca = PCA(n_components=2)
                    X_pca = pca.fit_transform(X_scaled)
                    
                    # 创建结果数据框
                    pca_df = pd.DataFrame({
                        'PC1': X_pca[:, 0],
                        'PC2': X_pca[:, 1],
                        'Color': data[color_var]
                    })
                    
                    # 绘制散点图
                    fig, ax = plt.subplots(figsize=(10, 8))
                    sns.scatterplot(data=pca_df, x='PC1', y='PC2', hue='Color', ax=ax)
                    ax.set_title("PCA降维结果")
                    ax.set_xlabel(f"主成分1 ({pca.explained_variance_ratio_[0]:.2%})")
                    ax.set_ylabel(f"主成分2 ({pca.explained_variance_ratio_[1]:.2%})")
                    st.pyplot(fig)
                    
                    # 显示解释方差比例
                    st.write(f"主成分1解释方差比例: {pca.explained_variance_ratio_[0]:.2%}")
                    st.write(f"主成分2解释方差比例: {pca.explained_variance_ratio_[1]:.2%}")
                    st.write(f"累计解释方差比例: {sum(pca.explained_variance_ratio_[:2]):.2%}")
                
                elif dim_reduction_method == "t-SNE":
                    # 执行t-SNE
                    tsne = TSNE(n_components=2, random_state=42)
                    X_tsne = tsne.fit_transform(X_scaled)
                    
                    # 创建结果数据框
                    tsne_df = pd.DataFrame({
                        'TSNE1': X_tsne[:, 0],
                        'TSNE2': X_tsne[:, 1],
                        'Color': data[color_var]
                    })
                    
                    # 绘制散点图
                    fig, ax = plt.subplots(figsize=(10, 8))
                    sns.scatterplot(data=tsne_df, x='TSNE1', y='TSNE2', hue='Color', ax=ax)
                    ax.set_title("t-SNE降维结果")
                    ax.set_xlabel("t-SNE维度1")
                    ax.set_ylabel("t-SNE维度2")
                    st.pyplot(fig)
    
    # 表达模式可视化
    elif viz_type == "表达模式可视化":
        st.markdown("### 表达模式可视化")
        
        # 选择可视化类型
        expr_viz_type = st.radio(
            "选择表达模式可视化类型",
            ["热图", "聚类热图", "表达差异箱线图"],
            index=0
        )
        
        if expr_viz_type == "热图":
            # 选择要可视化的基因数量
            n_genes = st.slider("选择要可视化的基因数量", 10, 50, 20)
            
            # 随机选择基因
            selected_genes = st.multiselect(
                "选择要可视化的基因",
                gene_cols,
                default=gene_cols[:min(n_genes, len(gene_cols))]
            )
            
            if selected_genes:
                # 准备数据
                gene_data = data[selected_genes]
                
                # 绘制热图
                fig, ax = plt.subplots(figsize=(12, 10))
                sns.heatmap(gene_data.T, cmap="viridis", ax=ax)
                ax.set_title("基因表达热图")
                ax.set_xlabel("样本")
                ax.set_ylabel("基因")
                st.pyplot(fig)
        
        elif expr_viz_type == "聚类热图":
            # 选择要可视化的基因数量
            n_genes = st.slider("选择要可视化的基因数量", 10, 50, 20)
            
            # 随机选择基因
            selected_genes = st.multiselect(
                "选择要可视化的基因",
                gene_cols,
                default=gene_cols[:min(n_genes, len(gene_cols))]
            )
            
            if selected_genes:
                # 准备数据
                gene_data = data[selected_genes]
                
                # 绘制聚类热图
                fig, ax = plt.subplots(figsize=(12, 10))
                sns.clustermap(gene_data.T, cmap="viridis", figsize=(12, 10))
                st.pyplot(fig)
        
        elif expr_viz_type == "表达差异箱线图":
            # 选择分组变量
            group_var = st.selectbox(
                "选择分组变量",
                clinical_cols,
                index=clinical_cols.index('disease_status') if 'disease_status' in clinical_cols else 0
            )
            
            # 选择要可视化的基因
            n_genes = st.slider("选择要可视化的基因数量", 3, 10, 5)
            
            selected_genes = st.multiselect(
                "选择要可视化的基因",
                gene_cols,
                default=gene_cols[:min(n_genes, len(gene_cols))]
            )
            
            if selected_genes:
                # 准备数据
                for gene in selected_genes:
                    # 创建数据框
                    plot_data = pd.DataFrame({
                        'Expression': data[gene],
                        'Group': data[group_var]
                    })
                    
                    # 绘制箱线图
                    fig, ax = plt.subplots(figsize=(10, 6))
                    sns.boxplot(data=plot_data, x='Group', y='Expression', ax=ax)
                    ax.set_title(f"{gene} 在不同组别中的表达")
                    ax.set_xlabel(group_var)
                    ax.set_ylabel("表达量")
                    st.pyplot(fig)
    
    # 交互式可视化
    elif viz_type == "交互式可视化":
        st.markdown("### 交互式可视化")
        
        # 选择可视化类型
        interactive_viz_type = st.radio(
            "选择交互式可视化类型",
            ["散点图", "箱线图", "热图", "3D散点图"],
            index=0
        )
        
        if interactive_viz_type == "散点图":
            # 选择X轴和Y轴变量
            x_var = st.selectbox("选择X轴变量", data.columns.tolist())
            y_var = st.selectbox("选择Y轴变量", [col for col in data.columns if col != x_var])
            
            # 选择颜色变量
            color_var = st.selectbox(
                "选择颜色变量",
                [None] + [col for col in data.columns if col not in [x_var, y_var]],
                index=0
            )
            
            # 创建散点图
            if color_var:
                fig = px.scatter(data, x=x_var, y=y_var, color=color_var, 
                                hover_data=[col for col in clinical_cols if col not in [x_var, y_var, color_var]][:5])
            else:
                fig = px.scatter(data, x=x_var, y=y_var, 
                                hover_data=[col for col in clinical_cols if col not in [x_var, y_var]][:5])
            
            fig.update_layout(title=f"{x_var} vs {y_var} 散点图", 
                             xaxis_title=x_var, 
                             yaxis_title=y_var)
            
            st.plotly_chart(fig)
        
        elif interactive_viz_type == "箱线图":
            # 选择Y轴变量
            y_var = st.selectbox("选择Y轴变量", data.columns.tolist())
            
            # 选择X轴分组变量
            categorical_cols = [col for col in clinical_cols 
                               if pd.api.types.is_categorical_dtype(data[col]) or 
                               pd.api.types.is_object_dtype(data[col]) or
                               (pd.api.types.is_numeric_dtype(data[col]) and data[col].nunique() < 10)]
            
            x_var = st.selectbox(
                "选择X轴分组变量",
                categorical_cols,
                index=0 if categorical_cols else None
            )
            
            if x_var:
                # 创建箱线图
                fig = px.box(data, x=x_var, y=y_var, color=x_var)
                
                fig.update_layout(title=f"{y_var} 在不同 {x_var} 组别中的分布", 
                                 xaxis_title=x_var, 
                                 yaxis_title=y_var)
                
                st.plotly_chart(fig)
        
        elif interactive_viz_type == "热图":
            # 选择要可视化的基因数量
            n_genes = st.slider("选择要可视化的基因数量", 5, 20, 10)
            
            # 随机选择基因
            selected_genes = st.multiselect(
                "选择要可视化的基因",
                gene_cols,
                default=gene_cols[:min(n_genes, len(gene_cols))]
            )
            
            if selected_genes:
                # 准备数据
                gene_data = data[selected_genes]
                
                # 创建热图
                fig = px.imshow(gene_data.T, 
                               labels=dict(x="样本", y="基因", color="表达量"),
                               x=gene_data.index,
                               y=selected_genes)
                
                fig.update_layout(title="基因表达热图")
                
                st.plotly_chart(fig)
        
        elif interactive_viz_type == "3D散点图":
            # 选择X、Y、Z轴变量
            x_var = st.selectbox("选择X轴变量", data.columns.tolist())
            y_var = st.selectbox("选择Y轴变量", [col for col in data.columns if col != x_var])
            z_var = st.selectbox("选择Z轴变量", [col for col in data.columns if col not in [x_var, y_var]])
            
            # 选择颜色变量
            color_var = st.selectbox(
                "选择颜色变量",
                [None] + [col for col in data.columns if col not in [x_var, y_var, z_var]],
                index=0
            )
            
            # 创建3D散点图
            if color_var:
                fig = px.scatter_3d(data, x=x_var, y=y_var, z=z_var, color=color_var,
                                   hover_data=[col for col in clinical_cols if col not in [x_var, y_var, z_var, color_var]][:3])
            else:
                fig = px.scatter_3d(data, x=x_var, y=y_var, z=z_var,
                                   hover_data=[col for col in clinical_cols if col not in [x_var, y_var, z_var]][:3])
            
            fig.update_layout(title=f"3D散点图: {x_var}, {y_var}, {z_var}")
            
            st.plotly_chart(fig)
    
    # 添加技术说明
    with st.expander("技术说明"):
        st.markdown("""
        #### 数据可视化的重要性
        
        数据可视化是RNA-seq数据分析中的关键步骤，它可以帮助研究人员：
        
        1. **探索数据特征**：了解数据的分布、趋势和模式
        2. **识别异常值**：发现可能影响分析结果的异常样本或基因
        3. **发现关系**：揭示基因之间或基因与临床特征之间的关系
        4. **展示结果**：以直观方式呈现分析结果
        
        #### 可视化类型
        
        1. **基本统计可视化**：
           - 分类变量分布：条形图、饼图等，展示不同类别的频率
           - 数值变量分布：直方图、箱线图等，展示数值分布特征
           - 基因表达分布：展示基因表达水平的分布情况
        
        2. **相关性分析**：
           - 热图：直观展示变量间的相关性强度
           - 散点图矩阵：展示变量两两之间的关系
        
        3. **降维可视化**：
           - PCA (主成分分析)：线性降维方法，保留数据中的主要变异
           - t-SNE：非线性降维方法，适合保留局部结构
        
        4. **表达模式可视化**：
           - 热图：展示基因在不同样本中的表达模式
           - 聚类热图：通过聚类算法揭示基因表达的相似性模式
           - 表达差异箱线图：比较不同组别间基因表达的差异
        
        5. **交互式可视化**：
           - 使用Plotly创建交互式图表，支持缩放、悬停查看详情等功能
           - 3D散点图：在三维空间中展示数据关系
        """)

if __name__ == "__main__":
    data_visualization_page()
