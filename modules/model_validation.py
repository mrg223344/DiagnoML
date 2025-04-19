import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import roc_curve, auc, precision_recall_curve, confusion_matrix
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
from sklearn.calibration import calibration_curve
import plotly.express as px
import plotly.graph_objects as go

def model_validation_page():
    st.title("模型验证")
    
    st.markdown("""
    ### 模型验证模块
    
    本模块提供全面的模型验证功能，帮助您评估机器学习模型的性能和临床应用价值。
    
    支持的验证方法：
    
    1. **ROC曲线分析**：评估模型的区分能力
    2. **校准曲线**：评估预测概率的准确性
    3. **决策曲线分析(DCA)**：评估模型在特定阈值下的临床净收益
    4. **外部验证**：使用外部数据集验证模型性能
    
    请先训练机器学习模型，然后使用本模块进行模型验证。
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
        # 使用测试集进行验证
        data = st.session_state['test_data']
        st.info("使用测试集数据进行模型验证")
    else:
        # 使用全部数据
        data = st.session_state['data']
        st.info("使用全部数据进行模型验证")
    
    # 准备特征和目标变量
    # 分离基因表达数据和临床数据
    gene_cols = [col for col in data.columns if col.startswith('GENE_')]
    clinical_cols = [col for col in data.columns if col not in gene_cols]
    
    # 选择目标变量
    target_col = st.selectbox(
        "选择目标变量",
        clinical_cols,
        index=clinical_cols.index('disease_status') if 'disease_status' in clinical_cols else 0,
        help="选择用于验证的目标变量，通常为分类变量"
    )
    
    # 准备特征和目标变量
    X = data[selected_features]
    y = data[target_col]
    
    # 检查是否为二分类问题
    is_binary = len(np.unique(y)) == 2
    
    if not is_binary:
        st.warning("当前仅支持二分类问题的模型验证")
    
    # 选择验证方法
    st.subheader("选择验证方法")
    
    validation_method = st.radio(
        "验证方法",
        ["ROC曲线分析", "校准曲线", "决策曲线分析(DCA)", "外部验证"],
        index=0
    )
    
    # ROC曲线分析
    if validation_method == "ROC曲线分析":
        st.markdown("### ROC曲线分析")
        
        if st.button("生成ROC曲线"):
            with st.spinner("正在生成ROC曲线..."):
                # 获取预测概率
                y_prob = model.predict_proba(X)[:, 1]
                
                # 计算ROC曲线
                fpr, tpr, thresholds = roc_curve(y, y_prob)
                roc_auc = auc(fpr, tpr)
                
                # 计算精确率-召回率曲线
                precision, recall, _ = precision_recall_curve(y, y_prob)
                pr_auc = auc(recall, precision)
                
                # 创建ROC曲线图
                fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))
                
                # ROC曲线
                ax1.plot(fpr, tpr, label=f'ROC曲线 (AUC = {roc_auc:.3f})')
                ax1.plot([0, 1], [0, 1], 'k--')
                ax1.set_xlim([0.0, 1.0])
                ax1.set_ylim([0.0, 1.05])
                ax1.set_xlabel('假阳性率 (1 - 特异性)')
                ax1.set_ylabel('真阳性率 (敏感性)')
                ax1.set_title('接收者操作特征曲线')
                ax1.legend(loc="lower right")
                
                # 精确率-召回率曲线
                ax2.plot(recall, precision, label=f'PR曲线 (AUC = {pr_auc:.3f})')
                ax2.set_xlim([0.0, 1.0])
                ax2.set_ylim([0.0, 1.05])
                ax2.set_xlabel('召回率')
                ax2.set_ylabel('精确率')
                ax2.set_title('精确率-召回率曲线')
                ax2.legend(loc="lower left")
                
                plt.tight_layout()
                st.pyplot(fig)
                
                # 显示ROC曲线的阈值和对应的敏感性、特异性
                st.subheader("ROC曲线阈值分析")
                
                # 创建阈值分析表格
                threshold_df = pd.DataFrame({
                    '阈值': thresholds,
                    '敏感性': tpr,
                    '特异性': 1 - fpr
                })
                
                # 计算约登指数 (Youden's J statistic)
                threshold_df['约登指数'] = threshold_df['敏感性'] + threshold_df['特异性'] - 1
                
                # 找到最佳阈值（约登指数最大）
                best_idx = threshold_df['约登指数'].idxmax()
                best_threshold = threshold_df.loc[best_idx, '阈值']
                
                st.write(f"最佳阈值 (约登指数最大): {best_threshold:.3f}")
                st.write(f"对应敏感性: {threshold_df.loc[best_idx, '敏感性']:.3f}")
                st.write(f"对应特异性: {threshold_df.loc[best_idx, '特异性']:.3f}")
                
                # 显示不同阈值下的性能
                st.write("不同阈值下的性能:")
                
                # 选择一些代表性阈值
                selected_indices = np.linspace(0, len(threshold_df)-1, 10, dtype=int)
                st.dataframe(threshold_df.iloc[selected_indices].round(3))
                
                # 使用最佳阈值计算混淆矩阵
                y_pred = (y_prob >= best_threshold).astype(int)
                cm = confusion_matrix(y, y_pred)
                
                # 绘制混淆矩阵
                fig, ax = plt.subplots(figsize=(8, 6))
                sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', ax=ax)
                ax.set_xlabel('预测标签')
                ax.set_ylabel('真实标签')
                ax.set_title('混淆矩阵 (最佳阈值)')
                st.pyplot(fig)
                
                # 计算性能指标
                accuracy = accuracy_score(y, y_pred)
                precision = precision_score(y, y_pred)
                recall = recall_score(y, y_pred)
                f1 = f1_score(y, y_pred)
                
                # 显示性能指标
                st.subheader("性能指标 (最佳阈值)")
                
                col1, col2 = st.columns(2)
                
                with col1:
                    st.write(f"准确率: {accuracy:.3f}")
                    st.write(f"精确率: {precision:.3f}")
                
                with col2:
                    st.write(f"召回率: {recall:.3f}")
                    st.write(f"F1分数: {f1:.3f}")
    
    # 校准曲线
    elif validation_method == "校准曲线":
        st.markdown("### 校准曲线")
        
        st.info("校准曲线展示了预测概率与实际概率的关系，理想情况下应该接近对角线。")
        
        # 选择分箱数量
        n_bins = st.slider("分箱数量", 5, 20, 10)
        
        if st.button("生成校准曲线"):
            with st.spinner("正在生成校准曲线..."):
                # 获取预测概率
                y_prob = model.predict_proba(X)[:, 1]
                
                # 计算校准曲线
                prob_true, prob_pred = calibration_curve(y, y_prob, n_bins=n_bins)
                
                # 创建校准曲线图
                fig, ax = plt.subplots(figsize=(10, 8))
                
                # 绘制校准曲线
                ax.plot(prob_pred, prob_true, marker='o', linewidth=1, label='校准曲线')
                
                # 绘制理想校准曲线（对角线）
                ax.plot([0, 1], [0, 1], linestyle='--', label='理想校准')
                
                # 绘制直方图
                ax.hist(y_prob, range=(0, 1), bins=n_bins, histtype='step', density=True, label='预测概率分布')
                
                ax.set_xlabel('预测概率')
                ax.set_ylabel('实际概率')
                ax.set_title('校准曲线')
                ax.legend()
                
                st.pyplot(fig)
                
                # 计算Brier分数
                from sklearn.metrics import brier_score_loss
                brier_score = brier_score_loss(y, y_prob)
                
                st.write(f"Brier分数: {brier_score:.4f} (越小越好)")
                
                # 显示校准曲线数据
                st.subheader("校准曲线数据")
                
                calibration_df = pd.DataFrame({
                    '预测概率': prob_pred,
                    '实际概率': prob_true
                })
                
                st.dataframe(calibration_df.round(3))
    
    # 决策曲线分析
    elif validation_method == "决策曲线分析(DCA)":
        st.markdown("### 决策曲线分析(DCA)")
        
        st.info("决策曲线分析评估了模型在不同阈值下的临床净收益，帮助确定模型的临床应用价值。")
        
        if st.button("生成决策曲线"):
            with st.spinner("正在生成决策曲线..."):
                # 获取预测概率
                y_prob = model.predict_proba(X)[:, 1]
                
                # 定义阈值范围
                thresholds = np.arange(0.01, 1.0, 0.01)
                
                # 计算净收益
                net_benefit_model = []
                net_benefit_all = []
                net_benefit_none = []
                
                for threshold in thresholds:
                    # 根据阈值进行预测
                    y_pred = (y_prob >= threshold).astype(int)
                    
                    # 计算真阳性和假阳性
                    tp = np.sum((y_pred == 1) & (y == 1))
                    fp = np.sum((y_pred == 1) & (y == 0))
                    
                    # 计算净收益
                    n = len(y)
                    net_benefit_model.append((tp - fp * (threshold / (1 - threshold))) / n)
                    net_benefit_all.append((np.sum(y) / n) - (0 * (threshold / (1 - threshold))))
                    net_benefit_none.append(0)
                
                # 创建决策曲线图
                fig, ax = plt.subplots(figsize=(10, 8))
                
                # 绘制模型的净收益曲线
                ax.plot(thresholds, net_benefit_model, label='模型', linewidth=2)
                
                # 绘制"全部治疗"的净收益曲线
                ax.plot(thresholds, net_benefit_all, label='全部治疗', linewidth=2, linestyle='--')
                
                # 绘制"无人治疗"的净收益曲线
                ax.plot(thresholds, net_benefit_none, label='无人治疗', linewidth=2, linestyle='--')
                
                ax.set_xlabel('阈值概率')
                ax.set_ylabel('净收益')
                ax.set_title('决策曲线分析')
                ax.legend()
                ax.grid(True)
                
                # 设置合适的y轴范围
                y_min = min(min(net_benefit_model), min(net_benefit_all), -0.05)
                y_max = max(max(net_benefit_model), max(net_benefit_all)) + 0.05
                ax.set_ylim([y_min, y_max])
                
                st.pyplot(fig)
                
                # 显示决策曲线数据
                st.subheader("决策曲线数据")
                
                dca_df = pd.DataFrame({
                    '阈值概率': thresholds,
                    '模型净收益': net_benefit_model,
                    '全部治疗净收益': net_benefit_all,
                    '无人治疗净收益': net_benefit_none
                })
                
                st.dataframe(dca_df.round(3))
    
    # 外部验证
    elif validation_method == "外部验证":
        st.markdown("### 外部验证")
        
        st.info("外部验证使用独立的数据集评估模型性能，是验证模型泛化能力的重要方法。")
        
        # 上传外部验证数据集
        uploaded_file = st.file_uploader("上传外部验证数据集 (CSV格式)", type=["csv"])
        
        if uploaded_file is not None:
            try:
                # 加载外部数据集
                external_data = pd.read_csv(uploaded_file)
                
                st.success(f"成功加载外部数据集: {uploaded_file.name}")
                
                # 显示数据预览
                st.subheader("外部数据集预览")
                st.dataframe(external_data.head())
                
                # 检查外部数据集是否包含所需特征和目标变量
                missing_features = [feature for feature in selected_features if feature not in external_data.columns]
                
                if missing_features:
                    st.error(f"外部数据集缺少以下特征: {', '.join(missing_features)}")
                    return
                
                if target_col not in external_data.columns:
                    st.error(f"外部数据集缺少目标变量: {target_col}")
                    return
                
                # 准备外部数据集的特征和目标变量
                X_external = external_data[selected_features]
                y_external = external_data[target_col]
                
                # 在外部数据集上进行预测
                y_prob_external = model.predict_proba(X_external)[:, 1]
                
                # 计算ROC曲线
                fpr, tpr, thresholds = roc_curve(y_external, y_prob_external)
                roc_auc = auc(fpr, tpr)
                
                # 创建ROC曲线图
                fig, ax = plt.subplots(figsize=(10, 8))
                
                # 绘制ROC曲线
                ax.plot(fpr, tpr, label=f'外部验证 ROC曲线 (AUC = {roc_auc:.3f})')
                ax.plot([0, 1], [0, 1], 'k--')
                ax.set_xlim([0.0, 1.0])
                ax.set_ylim([0.0, 1.05])
                ax.set_xlabel('假阳性率 (1 - 特异性)')
                ax.set_ylabel('真阳性率 (敏感性)')
                ax.set_title('外部验证 ROC曲线')
                ax.legend(loc="lower right")
                
                st.pyplot(fig)
                
                # 计算最佳阈值
                threshold_df = pd.DataFrame({
                    '阈值': thresholds,
                    '敏感性': tpr,
                    '特异性': 1 - fpr
                })
                
                # 计算约登指数
                threshold_df['约登指数'] = threshold_df['敏感性'] + threshold_df['特异性'] - 1
                
                # 找到最佳阈值
                best_idx = threshold_df['约登指数'].idxmax()
                best_threshold = threshold_df.loc[best_idx, '阈值']
                
                # 使用最佳阈值进行预测
                y_pred_external = (y_prob_external >= best_threshold).astype(int)
                
                # 计算性能指标
                accuracy = accuracy_score(y_external, y_pred_external)
                precision = precision_score(y_external, y_pred_external)
                recall = recall_score(y_external, y_pred_external)
                f1 = f1_score(y_external, y_pred_external)
                
                # 显示性能指标
                st.subheader("外部验证性能指标")
                
                col1, col2 = st.columns(2)
                
                with col1:
                    st.write(f"准确率: {accuracy:.3f}")
                    st.write(f"精确率: {precision:.3f}")
                
                with col2:
                    st.write(f"召回率: {recall:.3f}")
                    st.write(f"F1分数: {f1:.3f}")
                
                # 绘制混淆矩阵
                cm = confusion_matrix(y_external, y_pred_external)
                
                fig, ax = plt.subplots(figsize=(8, 6))
                sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', ax=ax)
                ax.set_xlabel('预测标签')
                ax.set_ylabel('真实标签')
                ax.set_title('外部验证混淆矩阵')
                st.pyplot(fig)
                
            except Exception as e:
                st.error(f"处理外部数据集时出错: {e}")
        else:
            st.warning("请上传外部验证数据集")
    
    # 添加技术说明
    with st.expander("技术说明"):
        st.markdown("""
        #### 模型验证的重要性
        
        模型验证是评估机器学习模型性能和临床应用价值的关键步骤：
        
        1. **评估泛化能力**：验证模型在新数据上的表现
        2. **避免过拟合**：确保模型不仅在训练数据上表现良好
        3. **指导临床决策**：评估模型在实际应用中的价值
        4. **比较不同模型**：提供客观的性能比较标准
        
        #### 验证方法
        
        1. **ROC曲线分析**：
           - 绘制不同阈值下的敏感性(真阳性率)和1-特异性(假阳性率)
           - AUC (曲线下面积) 是模型区分能力的综合指标，范围0.5-1.0，越高越好
           - 约登指数 (敏感性+特异性-1) 可用于确定最佳阈值
        
        2. **校准曲线**：
           - 评估预测概率与实际概率的一致性
           - 理想的校准曲线应接近对角线
           - Brier分数衡量预测概率的准确性，范围0-1，越低越好
        
        3. **决策曲线分析(DCA)**：
           - 评估模型在不同阈值下的临床净收益
           - 考虑了错误决策的不同代价
           - 帮助确定模型的临床应用价值和最佳决策阈值
        
        4. **外部验证**：
           - 使用独立的数据集评估模型性能
           - 是验证模型泛化能力的金标准
           - 外部验证性能通常低于内部验证，但更能反映实际应用价值
        """)

if __name__ == "__main__":
    model_validation_page()
