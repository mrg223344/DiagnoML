import streamlit as st
import pandas as pd
import numpy as np
from sklearn.impute import SimpleImputer, KNNImputer
from sklearn.preprocessing import StandardScaler, MinMaxScaler, OneHotEncoder, LabelEncoder
import matplotlib.pyplot as plt
import seaborn as sns

def data_preprocessing_page():
    st.title("数据预处理")
    
    st.markdown("""
    ### 数据预处理模块
    
    本模块提供全面的数据预处理功能，包括：
    
    1. **缺失值处理**：多种缺失值填充方法
    2. **分类变量编码**：将分类变量转换为数值形式
    3. **数据标准化/归一化**：使特征具有相似的尺度
    
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
    
    # 显示数据基本信息
    st.subheader("数据基本信息")
    
    # 显示缺失值信息
    missing_data = data[clinical_cols].isnull().sum()
    missing_data = missing_data[missing_data > 0]
    
    if not missing_data.empty:
        st.write("临床数据中的缺失值:")
        missing_df = pd.DataFrame({
            '变量': missing_data.index,
            '缺失值数量': missing_data.values,
            '缺失比例': (missing_data.values / len(data)).round(4) * 100
        })
        st.dataframe(missing_df)
    else:
        st.write("临床数据中没有缺失值")
    
    # 显示数据类型信息
    st.write("数据类型分布:")
    dtype_counts = data[clinical_cols].dtypes.value_counts()
    st.write(pd.DataFrame({
        '数据类型': dtype_counts.index.astype(str),
        '列数': dtype_counts.values
    }))
    
    # 预处理选项
    st.subheader("预处理选项")
    
    # 使用选项卡组织不同的预处理功能
    tab1, tab2, tab3 = st.tabs(["缺失值处理", "分类变量编码", "数据标准化/归一化"])
    
    # 缺失值处理
    with tab1:
        st.markdown("### 缺失值处理")
        
        # 选择要处理的列
        columns_with_missing = data.columns[data.isnull().any()].tolist()
        
        if not columns_with_missing:
            st.info("数据中没有缺失值，无需进行缺失值处理")
        else:
            selected_cols = st.multiselect(
                "选择要处理缺失值的列",
                columns_with_missing,
                default=columns_with_missing
            )
            
            if selected_cols:
                # 选择缺失值处理方法
                imputation_method = st.selectbox(
                    "选择缺失值填充方法",
                    ["均值填充", "中位数填充", "众数填充", "常数填充", "KNN填充"],
                    index=0
                )
                
                if imputation_method == "常数填充":
                    fill_value = st.text_input("填充值", "0")
                
                if st.button("执行缺失值处理"):
                    with st.spinner("正在处理缺失值..."):
                        # 创建数据副本
                        processed_data = data.copy()
                        
                        # 根据选择的方法处理缺失值
                        if imputation_method == "均值填充":
                            imputer = SimpleImputer(strategy='mean')
                            for col in selected_cols:
                                if pd.api.types.is_numeric_dtype(data[col]):
                                    processed_data[col] = imputer.fit_transform(data[col].values.reshape(-1, 1)).flatten()
                                else:
                                    st.warning(f"列 '{col}' 不是数值型，无法使用均值填充")
                        
                        elif imputation_method == "中位数填充":
                            imputer = SimpleImputer(strategy='median')
                            for col in selected_cols:
                                if pd.api.types.is_numeric_dtype(data[col]):
                                    processed_data[col] = imputer.fit_transform(data[col].values.reshape(-1, 1)).flatten()
                                else:
                                    st.warning(f"列 '{col}' 不是数值型，无法使用中位数填充")
                        
                        elif imputation_method == "众数填充":
                            imputer = SimpleImputer(strategy='most_frequent')
                            for col in selected_cols:
                                processed_data[col] = imputer.fit_transform(data[col].values.reshape(-1, 1)).flatten()
                        
                        elif imputation_method == "常数填充":
                            try:
                                # 尝试将填充值转换为数值
                                numeric_fill = float(fill_value)
                                imputer = SimpleImputer(strategy='constant', fill_value=numeric_fill)
                            except ValueError:
                                # 如果不是数值，则使用字符串
                                imputer = SimpleImputer(strategy='constant', fill_value=fill_value)
                            
                            for col in selected_cols:
                                processed_data[col] = imputer.fit_transform(data[col].values.reshape(-1, 1)).flatten()
                        
                        elif imputation_method == "KNN填充":
                            # KNN填充只能用于数值型数据
                            numeric_cols = [col for col in selected_cols if pd.api.types.is_numeric_dtype(data[col])]
                            if numeric_cols:
                                imputer = KNNImputer(n_neighbors=5)
                                processed_data[numeric_cols] = imputer.fit_transform(data[numeric_cols])
                            
                            non_numeric_cols = [col for col in selected_cols if col not in numeric_cols]
                            if non_numeric_cols:
                                st.warning(f"以下列不是数值型，无法使用KNN填充: {', '.join(non_numeric_cols)}")
                        
                        # 更新会话状态中的数据
                        st.session_state['data'] = processed_data
                        
                        # 显示处理结果
                        st.success("缺失值处理完成!")
                        
                        # 显示处理前后的缺失值对比
                        col1, col2 = st.columns(2)
                        
                        with col1:
                            st.write("处理前缺失值:")
                            missing_before = data[selected_cols].isnull().sum()
                            st.dataframe(pd.DataFrame({
                                '变量': missing_before.index,
                                '缺失值数量': missing_before.values
                            }))
                        
                        with col2:
                            st.write("处理后缺失值:")
                            missing_after = processed_data[selected_cols].isnull().sum()
                            st.dataframe(pd.DataFrame({
                                '变量': missing_after.index,
                                '缺失值数量': missing_after.values
                            }))
    
    # 分类变量编码
    with tab2:
        st.markdown("### 分类变量编码")
        
        # 识别分类变量
        categorical_cols = [col for col in clinical_cols 
                           if pd.api.types.is_object_dtype(data[col]) or 
                           pd.api.types.is_categorical_dtype(data[col])]
        
        if not categorical_cols:
            st.info("未检测到分类变量，无需进行编码")
        else:
            st.write(f"检测到 {len(categorical_cols)} 个分类变量:")
            
            # 显示分类变量的唯一值数量
            cat_info = pd.DataFrame({
                '变量': categorical_cols,
                '唯一值数量': [data[col].nunique() for col in categorical_cols]
            })
            st.dataframe(cat_info)
            
            # 选择要编码的列
            selected_cat_cols = st.multiselect(
                "选择要编码的分类变量",
                categorical_cols,
                default=categorical_cols
            )
            
            if selected_cat_cols:
                # 选择编码方法
                encoding_method = st.selectbox(
                    "选择编码方法",
                    ["One-Hot编码", "标签编码"],
                    index=0,
                    help="One-Hot编码将创建多个二进制列，标签编码将分类转换为单个数值列"
                )
                
                if st.button("执行分类变量编码"):
                    with st.spinner("正在编码分类变量..."):
                        # 创建数据副本
                        processed_data = data.copy()
                        
                        if encoding_method == "One-Hot编码":
                            # 使用pandas的get_dummies进行One-Hot编码
                            for col in selected_cat_cols:
                                # 创建One-Hot编码
                                dummies = pd.get_dummies(processed_data[col], prefix=col, drop_first=False)
                                
                                # 将编码后的列添加到数据中
                                processed_data = pd.concat([processed_data, dummies], axis=1)
                                
                                # 删除原始列
                                processed_data.drop(col, axis=1, inplace=True)
                            
                            st.success(f"One-Hot编码完成! 数据形状从 {data.shape} 变为 {processed_data.shape}")
                        
                        elif encoding_method == "标签编码":
                            # 使用LabelEncoder进行标签编码
                            label_encoders = {}
                            
                            for col in selected_cat_cols:
                                le = LabelEncoder()
                                processed_data[col] = le.fit_transform(processed_data[col].astype(str))
                                
                                # 保存编码器以便后续使用
                                label_encoders[col] = le
                                
                                # 显示编码映射
                                mapping = dict(zip(le.classes_, le.transform(le.classes_)))
                                st.write(f"{col} 编码映射:", mapping)
                            
                            # 保存标签编码器到会话状态
                            st.session_state['label_encoders'] = label_encoders
                            
                            st.success("标签编码完成!")
                        
                        # 更新会话状态中的数据
                        st.session_state['data'] = processed_data
                        
                        # 显示处理后的数据预览
                        st.subheader("编码后数据预览")
                        st.dataframe(processed_data.head())
    
    # 数据标准化/归一化
    with tab3:
        st.markdown("### 数据标准化/归一化")
        
        # 识别数值变量
        numeric_cols = [col for col in data.columns 
                       if pd.api.types.is_numeric_dtype(data[col]) and 
                       col != 'sample_id']  # 排除样本ID列
        
        # 默认选择基因表达列
        default_cols = gene_cols if gene_cols else numeric_cols[:10]
        
        # 选择要标准化的列
        selected_num_cols = st.multiselect(
            "选择要标准化/归一化的列",
            numeric_cols,
            default=default_cols
        )
        
        if selected_num_cols:
            # 选择标准化/归一化方法
            scaling_method = st.selectbox(
                "选择标准化/归一化方法",
                ["Z-score标准化 (StandardScaler)", "Min-Max归一化 (MinMaxScaler)"],
                index=0
            )
            
            if st.button("执行数据标准化/归一化"):
                with st.spinner("正在标准化/归一化数据..."):
                    # 创建数据副本
                    processed_data = data.copy()
                    
                    # 选择要处理的数据
                    X = processed_data[selected_num_cols]
                    
                    if scaling_method == "Z-score标准化 (StandardScaler)":
                        scaler = StandardScaler()
                        scaled_data = scaler.fit_transform(X)
                        
                        # 更新数据
                        processed_data[selected_num_cols] = scaled_data
                        
                        st.success("Z-score标准化完成!")
                        
                    elif scaling_method == "Min-Max归一化 (MinMaxScaler)":
                        scaler = MinMaxScaler()
                        scaled_data = scaler.fit_transform(X)
                        
                        # 更新数据
                        processed_data[selected_num_cols] = scaled_data
                        
                        st.success("Min-Max归一化完成!")
                    
                    # 保存标准化器到会话状态
                    st.session_state['scaler'] = scaler
                    
                    # 更新会话状态中的数据
                    st.session_state['data'] = processed_data
                    
                    # 可视化标准化/归一化前后的分布
                    st.subheader("标准化/归一化前后对比")
                    
                    # 选择一部分列进行可视化
                    vis_cols = selected_num_cols[:5]  # 最多显示5列
                    
                    fig, axes = plt.subplots(len(vis_cols), 2, figsize=(12, 3*len(vis_cols)))
                    
                    for i, col in enumerate(vis_cols):
                        # 原始数据分布
                        sns.histplot(data[col], kde=True, ax=axes[i, 0])
                        axes[i, 0].set_title(f"原始 {col}")
                        
                        # 标准化/归一化后的分布
                        sns.histplot(processed_data[col], kde=True, ax=axes[i, 1])
                        axes[i, 1].set_title(f"处理后 {col}")
                    
                    plt.tight_layout()
                    st.pyplot(fig)
                    
                    # 显示处理后的数据预览
                    st.subheader("处理后数据预览")
                    st.dataframe(processed_data.head())
    
    # 添加技术说明
    with st.expander("技术说明"):
        st.markdown("""
        #### 缺失值处理
        
        缺失值是数据分析中常见的问题，可能会影响模型的性能。常用的处理方法包括：
        
        - **均值填充**：用特征的平均值填充缺失值，适用于正态分布的数据
        - **中位数填充**：用特征的中位数填充缺失值，对异常值不敏感
        - **众数填充**：用特征的众数填充缺失值，适用于分类变量
        - **常数填充**：用指定的常数填充缺失值
        - **KNN填充**：基于K近邻算法，使用相似样本的值填充缺失值
        
        #### 分类变量编码
        
        机器学习算法通常要求输入为数值型数据，因此需要将分类变量转换为数值形式：
        
        - **One-Hot编码**：将每个类别转换为一个二进制特征，适用于类别之间没有顺序关系的情况
        - **标签编码**：将每个类别映射为一个整数，适用于类别之间有顺序关系的情况
        
        #### 数据标准化/归一化
        
        标准化和归一化可以使不同尺度的特征具有可比性，有助于提高模型性能：
        
        - **Z-score标准化**：将特征转换为均值为0、标准差为1的分布，适用于数据近似正态分布的情况
        - **Min-Max归一化**：将特征缩放到指定范围（通常是[0,1]），保留原始分布的形状
        """)

if __name__ == "__main__":
    data_preprocessing_page()
