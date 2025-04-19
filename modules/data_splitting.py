import streamlit as st
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split

def data_splitting_page():
    st.title("数据拆分")
    
    st.markdown("""
    ### 数据拆分模块
    
    本模块提供数据集拆分功能，可以将数据集划分为训练集和测试集，用于机器学习模型的训练和评估。
    
    您可以设置以下参数：
    
    1. **测试集比例**：测试集占总数据的比例
    2. **随机种子**：确保结果可重复性的随机数种子
    3. **分层抽样**：是否根据目标变量进行分层抽样
    
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
    st.write(f"样本数量: {data.shape[0]}")
    st.write(f"特征数量: {len(gene_cols)}")
    st.write(f"临床特征: {len(clinical_cols)}")
    
    # 数据拆分参数设置
    st.subheader("拆分参数设置")
    
    col1, col2 = st.columns(2)
    
    with col1:
        test_size = st.slider("测试集比例", 0.1, 0.5, 0.2, 0.05, 
                             help="测试集占总数据的比例，通常为20%-30%")
        random_state = st.number_input("随机种子", 0, 1000, 42, 
                                      help="设置随机种子以确保结果可重复")
    
    with col2:
        stratify_option = st.checkbox("使用分层抽样", True, 
                                     help="根据目标变量进行分层抽样，保持各类别比例一致")
        
        if stratify_option:
            stratify_col = st.selectbox("选择分层变量", 
                                       clinical_cols,
                                       index=clinical_cols.index('disease_status') if 'disease_status' in clinical_cols else 0,
                                       help="选择用于分层的目标变量，通常为分类变量")
        else:
            stratify_col = None
    
    # 执行数据拆分
    if st.button("执行数据拆分"):
        with st.spinner("正在拆分数据..."):
            # 准备分层变量
            stratify = data[stratify_col] if stratify_option and stratify_col else None
            
            # 执行拆分
            train_idx, test_idx = train_test_split(
                np.arange(len(data)),
                test_size=test_size,
                random_state=random_state,
                stratify=stratify
            )
            
            # 获取训练集和测试集
            train_data = data.iloc[train_idx].reset_index(drop=True)
            test_data = data.iloc[test_idx].reset_index(drop=True)
            
            # 保存到会话状态
            st.session_state['train_data'] = train_data
            st.session_state['test_data'] = test_data
            st.session_state['train_idx'] = train_idx
            st.session_state['test_idx'] = test_idx
            
            # 显示拆分结果
            st.success(f"数据拆分成功! 训练集: {len(train_data)}样本, 测试集: {len(test_data)}样本")
            
            # 显示训练集和测试集的分布
            if stratify_option and stratify_col:
                st.subheader("分层变量分布")
                
                col1, col2 = st.columns(2)
                
                with col1:
                    st.write("训练集分布")
                    train_dist = train_data[stratify_col].value_counts(normalize=True).reset_index()
                    train_dist.columns = [stratify_col, '比例']
                    st.dataframe(train_dist)
                
                with col2:
                    st.write("测试集分布")
                    test_dist = test_data[stratify_col].value_counts(normalize=True).reset_index()
                    test_dist.columns = [stratify_col, '比例']
                    st.dataframe(test_dist)
            
            # 显示训练集和测试集预览
            st.subheader("训练集预览")
            st.dataframe(train_data.head())
            
            st.subheader("测试集预览")
            st.dataframe(test_data.head())
    
    # 如果已经有拆分的数据，显示拆分信息
    elif 'train_data' in st.session_state and 'test_data' in st.session_state:
        st.success(f"数据已拆分! 训练集: {len(st.session_state['train_data'])}样本, 测试集: {len(st.session_state['test_data'])}样本")
        
        if st.checkbox("显示拆分数据预览"):
            st.subheader("训练集预览")
            st.dataframe(st.session_state['train_data'].head())
            
            st.subheader("测试集预览")
            st.dataframe(st.session_state['test_data'].head())
    
    # 添加技术说明
    with st.expander("技术说明"):
        st.markdown("""
        #### 数据拆分的重要性
        
        在机器学习中，将数据集拆分为训练集和测试集是一种标准做法，目的是：
        
        1. **训练集**：用于训练模型，模型会从这些数据中学习模式和关系
        2. **测试集**：用于评估模型性能，这些数据在训练过程中对模型是"不可见的"
        
        这种拆分可以帮助我们评估模型的泛化能力，即模型在新数据上的表现。
        
        #### 分层抽样
        
        分层抽样确保训练集和测试集中各类别的比例与原始数据集相同。这在处理不平衡数据集时尤为重要，可以避免某些类别在训练集或测试集中过度或不足表示。
        
        #### 随机种子
        
        设置随机种子可以确保数据拆分的可重复性，这对于实验结果的复现和比较非常重要。
        """)

if __name__ == "__main__":
    data_splitting_page()
