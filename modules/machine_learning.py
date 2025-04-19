import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.linear_model import LogisticRegression, Lasso, LassoCV
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import cross_val_score, StratifiedKFold
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score
from sklearn.metrics import confusion_matrix, classification_report, roc_curve, auc
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
import time

def machine_learning_page():
    st.title("机器学习模型")
    
    st.markdown("""
    ### 机器学习模型模块
    
    本模块提供两种建模模式：
    
    1. **自定义建模（Custom Mode）**：用户上传基因列表（如已知通路基因），平台自动匹配训练集生成预后模型
    2. **自动化建模（Auto Mode）**：
       - 单变量Cox回归初筛（p<0.05）
       - LASSO-Cox排除共线性基因
       - 随机生存森林（RSF）等优化特征组合，输出最优基因集
    
    支持多种分类模型，包括逻辑回归、随机森林、支持向量机等。
    
    请上传数据或使用示例数据进行操作。
    """)
    
    # 检查是否有数据加载
    if 'data' not in st.session_state:
        st.warning("请先在侧边栏上传数据或生成示例数据")
        return
    
    # 获取数据
    if 'train_data' in st.session_state and 'test_data' in st.session_state:
        # 如果已经拆分了数据，使用训练集和测试集
        train_data = st.session_state['train_data']
        test_data = st.session_state['test_data']
        st.info("使用已拆分的训练集和测试集进行建模")
    else:
        # 否则使用全部数据，并进行拆分
        st.warning("未检测到已拆分的训练集和测试集，将使用全部数据并进行自动拆分")
        from sklearn.model_selection import train_test_split
        
        data = st.session_state['data']
        # 分离特征和目标变量
        gene_cols = [col for col in data.columns if col.startswith('GENE_')]
        clinical_cols = [col for col in data.columns if col not in gene_cols]
        
        # 选择目标变量
        target_col = st.selectbox(
            "选择目标变量",
            clinical_cols,
            index=clinical_cols.index('disease_status') if 'disease_status' in clinical_cols else 0,
            help="选择用于建模的目标变量，通常为分类变量"
        )
        
        # 拆分数据
        X = data[gene_cols]
        y = data[target_col]
        
        train_X, test_X, train_y, test_y = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)
        
        # 创建训练集和测试集
        train_data = pd.concat([train_X, train_y], axis=1)
        test_data = pd.concat([test_X, test_y], axis=1)
        
        # 保存到会话状态
        st.session_state['train_data'] = train_data
        st.session_state['test_data'] = test_data
    
    # 分离基因表达数据和临床数据
    # 假设所有以GENE_开头的列是基因表达数据
    gene_cols = [col for col in train_data.columns if col.startswith('GENE_')]
    clinical_cols = [col for col in train_data.columns if col not in gene_cols]
    
    if len(gene_cols) == 0:
        st.error("未检测到基因表达数据列（以GENE_开头的列）")
        return
    
    # 选择目标变量
    target_col = st.selectbox(
        "选择目标变量",
        clinical_cols,
        index=clinical_cols.index('disease_status') if 'disease_status' in clinical_cols else 0,
        help="选择用于建模的目标变量，通常为分类变量"
    )
    
    # 检查目标变量类型
    is_classification = pd.api.types.is_categorical_dtype(train_data[target_col]) or train_data[target_col].nunique() < 10
    
    if not is_classification:
        st.warning(f"目标变量 {target_col} 可能不是分类变量，当前版本仅支持分类模型")
    
    # 显示数据基本信息
    st.subheader("数据基本信息")
    st.write(f"训练集样本数: {train_data.shape[0]}")
    st.write(f"测试集样本数: {test_data.shape[0]}")
    st.write(f"基因特征数量: {len(gene_cols)}")
    
    # 显示目标变量分布
    st.subheader("目标变量分布")
    
    # 计算训练集和测试集中的目标变量分布
    train_target_counts = train_data[target_col].value_counts()
    test_target_counts = test_data[target_col].value_counts()
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.write("训练集目标变量分布")
        st.dataframe(pd.DataFrame({
            '类别': train_target_counts.index,
            '样本数': train_target_counts.values,
            '百分比': (train_target_counts.values / train_target_counts.sum() * 100).round(2)
        }))
    
    with col2:
        st.write("测试集目标变量分布")
        st.dataframe(pd.DataFrame({
            '类别': test_target_counts.index,
            '样本数': test_target_counts.values,
            '百分比': (test_target_counts.values / test_target_counts.sum() * 100).round(2)
        }))
    
    # 建模模式选择
    st.subheader("建模模式")
    
    model_mode = st.radio(
        "选择建模模式",
        ["自定义建模 (Custom Mode)", "自动化建模 (Auto Mode)"],
        index=0
    )
    
    # 自定义建模模式
    if model_mode == "自定义建模 (Custom Mode)":
        st.markdown("### 自定义建模")
        
        # 特征选择
        st.subheader("特征选择")
        
        feature_selection_method = st.radio(
            "特征选择方法",
            ["使用所有基因特征", "使用特征工程选择的特征", "手动输入基因列表"],
            index=0
        )
        
        if feature_selection_method == "使用所有基因特征":
            selected_features = gene_cols
            st.info(f"将使用所有 {len(selected_features)} 个基因特征进行建模")
        
        elif feature_selection_method == "使用特征工程选择的特征":
            if 'selected_features' in st.session_state:
                selected_features = st.session_state['selected_features']
                st.success(f"将使用特征工程选择的 {len(selected_features)} 个特征进行建模")
            else:
                st.warning("未找到特征工程选择的特征，将使用所有基因特征")
                selected_features = gene_cols
        
        elif feature_selection_method == "手动输入基因列表":
            gene_list_input = st.text_area(
                "输入基因列表（每行一个基因名称）",
                height=150,
                help="输入已知的基因列表，平台将自动匹配数据集中的对应基因"
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
                    selected_features = matched_genes
                    st.success(f"成功匹配 {len(selected_features)} 个基因")
                else:
                    st.error("未能匹配任何基因，请检查输入的基因名称")
                    selected_features = gene_cols
            else:
                st.warning("未输入基因列表，将使用所有基因特征")
                selected_features = gene_cols
        
        # 模型选择
        st.subheader("模型选择")
        
        model_type = st.selectbox(
            "选择模型类型",
            ["逻辑回归", "随机森林", "支持向量机", "K近邻", "梯度提升树"],
            index=0,
            help="选择用于分类的机器学习模型"
        )
        
        # 模型参数设置
        st.subheader("模型参数")
        
        if model_type == "逻辑回归":
            C = st.slider("正则化强度倒数 (C)", 0.01, 10.0, 1.0, 0.01, help="较小的值表示更强的正则化")
            penalty = st.selectbox("正则化类型", ["l2", "l1", "elasticnet", "none"], index=0)
            solver = st.selectbox("优化算法", ["lbfgs", "liblinear", "newton-cg", "sag", "saga"], index=0)
            
            model_params = {
                "C": C,
                "penalty": penalty,
                "solver": solver,
                "random_state": 42
            }
        
        elif model_type == "随机森林":
            n_estimators = st.slider("树的数量", 10, 500, 100, 10)
            max_depth = st.slider("树的最大深度", 1, 50, 10, 1)
            min_samples_split = st.slider("内部节点分裂所需的最小样本数", 2, 20, 2, 1)
            min_samples_leaf = st.slider("叶节点所需的最小样本数", 1, 20, 1, 1)
            
            model_params = {
                "n_estimators": n_estimators,
                "max_depth": max_depth,
                "min_samples_split": min_samples_split,
                "min_samples_leaf": min_samples_leaf,
                "random_state": 42
            }
        
        elif model_type == "支持向量机":
            C = st.slider("正则化参数 (C)", 0.1, 10.0, 1.0, 0.1)
            kernel = st.selectbox("核函数", ["linear", "poly", "rbf", "sigmoid"], index=2)
            gamma = st.selectbox("核系数", ["scale", "auto"], index=0)
            
            model_params = {
                "C": C,
                "kernel": kernel,
                "gamma": gamma,
                "probability": True,
                "random_state": 42
            }
        
        elif model_type == "K近邻":
            n_neighbors = st.slider("邻居数量", 1, 20, 5, 1)
            weights = st.selectbox("权重函数", ["uniform", "distance"], index=0)
            metric = st.selectbox("距离度量", ["euclidean", "manhattan", "minkowski"], index=0)
            
            model_params = {
                "n_neighbors": n_neighbors,
                "weights": weights,
                "metric": metric
            }
        
        elif model_type == "梯度提升树":
            n_estimators = st.slider("提升迭代次数", 10, 500, 100, 10)
            learning_rate = st.slider("学习率", 0.01, 1.0, 0.1, 0.01)
            max_depth = st.slider("树的最大深度", 1, 10, 3, 1)
            subsample = st.slider("子样本比例", 0.1, 1.0, 1.0, 0.1)
            
            model_params = {
                "n_estimators": n_estimators,
                "learning_rate": learning_rate,
                "max_depth": max_depth,
                "subsample": subsample,
                "random_state": 42
            }
        
        # 交叉验证设置
        st.subheader("交叉验证设置")
        
        cv_folds = st.slider("交叉验证折数", 2, 10, 5, 1, help="K折交叉验证中的K值")
        
        # 训练模型
        if st.button("训练模型"):
            with st.spinner("正在训练模型..."):
                # 准备特征和目标变量
                X_train = train_data[selected_features]
                y_train = train_data[target_col]
                
                X_test = test_data[selected_features]
                y_test = test_data[target_col]
                
                # 创建模型
                if model_type == "逻辑回归":
                    model = LogisticRegression(**model_params)
                elif model_type == "随机森林":
                    model = RandomForestClassifier(**model_params)
                elif model_type == "支持向量机":
                    model = SVC(**model_params)
                elif model_type == "K近邻":
                    model = KNeighborsClassifier(**model_params)
                elif model_type == "梯度提升树":
                    model = GradientBoostingClassifier(**model_params)
                
                # 创建包含标准化的管道
                pipeline = Pipeline([
                    ('scaler', StandardScaler()),
                    ('model', model)
                ])
                
                # 交叉验证
                cv = StratifiedKFold(n_splits=cv_folds, shuffle=True, random_state=42)
                cv_scores = cross_val_score(pipeline, X_train, y_train, cv=cv, scoring='accuracy')
                
                st.success(f"交叉验证完成! 平均准确率: {cv_scores.mean():.4f} ± {cv_scores.std():.4f}")
                
                # 在整个训练集上训练模型
                pipeline.fit(X_train, y_train)
                
                # 在测试集上评估模型
                y_pred = pipeline.predict(X_test)
                y_prob = pipeline.predict_proba(X_test)
                
                # 计算评估指标
                accuracy = accuracy_score(y_test, y_pred)
                
                # 保存模型
                st.session_state['model'] = pipeline
                st.session_state['model_features'] = selected_features
                
                # 显示评估结果
                st.subheader("模型评估")
                
                st.write(f"测试集准确率: {accuracy:.4f}")
                
                # 如果是二分类问题
                if len(np.unique(y_train)) == 2:
                    precision = precision_score(y_test, y_pred, average='binary')
                    recall = recall_score(y_test, y_pred, average='binary')
                    f1 = f1_score(y_test, y_pred, average='binary')
                    
                    st.write(f"精确率: {precision:.4f}")
                    st.write(f"召回率: {recall:.4f}")
                    st.write(f"F1分数: {f1:.4f}")
                    
                    # 计算ROC曲线
                    fpr, tpr, _ = roc_curve(y_test, y_prob[:, 1])
                    roc_auc = auc(fpr, tpr)
                    
                    # 绘制ROC曲线
                    fig, ax = plt.subplots(figsize=(8, 6))
                    ax.plot(fpr, tpr, label=f'ROC曲线 (AUC = {roc_auc:.4f})')
                    ax.plot([0, 1], [0, 1], 'k--')
                    ax.set_xlim([0.0, 1.0])
                    ax.set_ylim([0.0, 1.05])
                    ax.set_xlabel('假阳性率')
                    ax.set_ylabel('真阳性率')
                    ax.set_title('接收者操作特征曲线')
                    ax.legend(loc="lower right")
                    st.pyplot(fig)
                
                # 混淆矩阵
                cm = confusion_matrix(y_test, y_pred)
                
                fig, ax = plt.subplots(figsize=(8, 6))
                sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', ax=ax)
                ax.set_xlabel('预测标签')
                ax.set_ylabel('真实标签')
                ax.set_title('混淆矩阵')
                st.pyplot(fig)
                
                # 分类报告
                report = classification_report(y_test, y_pred, output_dict=True)
                report_df = pd.DataFrame(report).transpose()
                st.write("分类报告:")
                st.dataframe(report_df)
                
                # 特征重要性（如果模型支持）
                if hasattr(model, 'feature_importances_') or hasattr(model, 'coef_'):
                    st.subheader("特征重要性")
                    
                    if hasattr(model, 'feature_importances_'):
                        # 对于随机森林和梯度提升树
                        importances = pipeline.named_steps['model'].feature_importances_
                    else:
                        # 对于逻辑回归
                        importances = np.abs(pipeline.named_steps['model'].coef_[0])
                    
                    # 创建特征重要性数据框
                    feature_importance_df = pd.DataFrame({
                        'Feature': selected_features,
                        'Importance': importances
                    })
                    
                    # 按重要性排序
                    feature_importance_df = feature_importance_df.sort_values('Importance', ascending=False)
                    
                    # 显示前20个特征的重要性
                    top_features = feature_importance_df.head(min(20, len(selected_features)))
                    
                    # 绘制特征重要性条形图
                    fig, ax = plt.subplots(figsize=(10, 8))
                    sns.barplot(x='Importance', y='Feature', data=top_features, ax=ax)
                    ax.set_title("特征重要性")
                    st.pyplot(fig)
                    
                    # 显示特征重要性表格
                    st.dataframe(top_features)
    
    # 自动化建模模式
    else:
        st.markdown("### 自动化建模")
        
        st.info("自动化建模将执行以下步骤：\n1. 单变量筛选\n2. LASSO特征选择\n3. 随机森林优化")
        
        # 单变量筛选参数
        st.subheader("单变量筛选参数")
        
        p_threshold = st.slider(
            "p值阈值",
            0.001, 0.1, 0.05, 0.001,
            help="用于初步筛选特征的p值阈值，小于该阈值的特征将被保留"
        )
        
        # LASSO参数
        st.subheader("LASSO参数")
        
        alpha_min_ratio = st.slider(
            "最小alpha比例",
            0.0001, 0.1, 0.001, 0.0001,
            help="LASSO正则化参数的最小值与最大值的比例"
        )
        
        n_alphas = st.slider(
            "alpha数量",
            50, 200, 100, 10,
            help="LASSO正则化参数的数量"
        )
        
        # 随机森林参数
        st.subheader("随机森林参数")
        
        n_estimators = st.slider("树的数量", 50, 500, 100, 10)
        max_features = st.slider("每棵树考虑的最大特征数", 5, 50, 20, 5)
        
        # 执行自动化建模
        if st.button("执行自动化建模"):
            with st.spinner("正在执行自动化建模..."):
                # 准备特征和目标变量
                X_train = train_data[gene_cols]
                y_train = train_data[target_col]
                
                X_test = test_data[gene_cols]
                y_test = test_data[target_col]
                
                # 步骤1: 单变量筛选
                st.write("步骤1: 单变量筛选")
                
                from sklearn.feature_selection import SelectKBest, f_classif
                
                # 使用F检验进行单变量特征选择
                selector = SelectKBest(f_classif, k='all')
                selector.fit(X_train, y_train)
                
                # 获取p值
                p_values = selector.pvalues_
                
                # 根据p值筛选特征
                selected_features_univariate = [gene_cols[i] for i in range(len(gene_cols)) if p_values[i] < p_threshold]
                
                st.write(f"单变量筛选后保留 {len(selected_features_univariate)} 个特征")
                
                # 如果没有特征被选中，使用所有特征
                if len(selected_features_univariate) == 0:
                    st.warning("单变量筛选未选中任何特征，将使用所有特征")
                    selected_features_univariate = gene_cols
                
                # 步骤2: LASSO特征选择
                st.write("步骤2: LASSO特征选择")
                
                # 准备数据
                X_train_univariate = X_train[selected_features_univariate]
                X_test_univariate = X_test[selected_features_univariate]
                
                # 标准化特征
                scaler = StandardScaler()
                X_train_scaled = scaler.fit_transform(X_train_univariate)
                X_test_scaled = scaler.transform(X_test_univariate)
                
                # 使用交叉验证确定最优alpha
                lasso_cv = LassoCV(
                    cv=5,
                    random_state=42,
                    max_iter=10000,
                    alphas=np.logspace(np.log10(0.1), np.log10(0.1 * alpha_min_ratio), n_alphas)
                )
                
                # 拟合模型
                lasso_cv.fit(X_train_scaled, y_train)
                
                # 获取最优alpha
                best_alpha = lasso_cv.alpha_
                st.write(f"LASSO最优alpha: {best_alpha:.6f}")
                
                # 使用最优alpha训练LASSO模型
                lasso = Lasso(alpha=best_alpha, random_state=42, max_iter=10000)
                lasso.fit(X_train_scaled, y_train)
                
                # 获取非零系数的特征
                selected_indices = np.where(lasso.coef_ != 0)[0]
                selected_features_lasso = [selected_features_univariate[i] for i in selected_indices]
                
                st.write(f"LASSO特征选择后保留 {len(selected_features_lasso)} 个特征")
                
                # 如果没有特征被选中，使用单变量筛选的特征
                if len(selected_features_lasso) == 0:
                    st.warning("LASSO未选中任何特征，将使用单变量筛选的特征")
                    selected_features_lasso = selected_features_univariate
                
                # 步骤3: 随机森林优化
                st.write("步骤3: 随机森林优化")
                
                # 准备数据
                X_train_lasso = X_train[selected_features_lasso]
                X_test_lasso = X_test[selected_features_lasso]
                
                # 训练随机森林模型
                rf = RandomForestClassifier(
                    n_estimators=n_estimators,
                    max_features=min(max_features, len(selected_features_lasso)),
                    random_state=42
                )
                
                rf.fit(X_train_lasso, y_train)
                
                # 获取特征重要性
                importances = rf.feature_importances_
                
                # 创建特征重要性数据框
                feature_importance_df = pd.DataFrame({
                    'Feature': selected_features_lasso,
                    'Importance': importances
                })
                
                # 按重要性排序
                feature_importance_df = feature_importance_df.sort_values('Importance', ascending=False)
                
                # 选择前20个最重要的特征
                top_features = feature_importance_df.head(min(20, len(selected_features_lasso)))
                selected_features_rf = top_features['Feature'].tolist()
                
                st.write(f"随机森林优化后选择 {len(selected_features_rf)} 个最重要的特征")
                
                # 使用最终选择的特征训练逻辑回归模型
                X_train_final = X_train[selected_features_rf]
                X_test_final = X_test[selected_features_rf]
                
                # 创建包含标准化的管道
                pipeline = Pipeline([
                    ('scaler', StandardScaler()),
                    ('model', LogisticRegression(random_state=42))
                ])
                
                # 训练模型
                pipeline.fit(X_train_final, y_train)
                
                # 在测试集上评估模型
                y_pred = pipeline.predict(X_test_final)
                y_prob = pipeline.predict_proba(X_test_final)
                
                # 计算评估指标
                accuracy = accuracy_score(y_test, y_pred)
                
                # 保存模型和特征
                st.session_state['model'] = pipeline
                st.session_state['model_features'] = selected_features_rf
                
                # 显示评估结果
                st.subheader("模型评估")
                
                st.write(f"测试集准确率: {accuracy:.4f}")
                
                # 如果是二分类问题
                if len(np.unique(y_train)) == 2:
                    precision = precision_score(y_test, y_pred, average='binary')
                    recall = recall_score(y_test, y_pred, average='binary')
                    f1 = f1_score(y_test, y_pred, average='binary')
                    
                    st.write(f"精确率: {precision:.4f}")
                    st.write(f"召回率: {recall:.4f}")
                    st.write(f"F1分数: {f1:.4f}")
                    
                    # 计算ROC曲线
                    fpr, tpr, _ = roc_curve(y_test, y_prob[:, 1])
                    roc_auc = auc(fpr, tpr)
                    
                    # 绘制ROC曲线
                    fig, ax = plt.subplots(figsize=(8, 6))
                    ax.plot(fpr, tpr, label=f'ROC曲线 (AUC = {roc_auc:.4f})')
                    ax.plot([0, 1], [0, 1], 'k--')
                    ax.set_xlim([0.0, 1.0])
                    ax.set_ylim([0.0, 1.05])
                    ax.set_xlabel('假阳性率')
                    ax.set_ylabel('真阳性率')
                    ax.set_title('接收者操作特征曲线')
                    ax.legend(loc="lower right")
                    st.pyplot(fig)
                
                # 混淆矩阵
                cm = confusion_matrix(y_test, y_pred)
                
                fig, ax = plt.subplots(figsize=(8, 6))
                sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', ax=ax)
                ax.set_xlabel('预测标签')
                ax.set_ylabel('真实标签')
                ax.set_title('混淆矩阵')
                st.pyplot(fig)
                
                # 分类报告
                report = classification_report(y_test, y_pred, output_dict=True)
                report_df = pd.DataFrame(report).transpose()
                st.write("分类报告:")
                st.dataframe(report_df)
                
                # 显示最终选择的特征
                st.subheader("最终选择的特征")
                
                # 绘制特征重要性条形图
                fig, ax = plt.subplots(figsize=(10, 8))
                sns.barplot(x='Importance', y='Feature', data=top_features, ax=ax)
                ax.set_title("特征重要性")
                st.pyplot(fig)
                
                # 显示特征重要性表格
                st.dataframe(top_features)
    
    # 添加技术说明
    with st.expander("技术说明"):
        st.markdown("""
        #### 自定义建模与自动化建模
        
        1. **自定义建模 (Custom Mode)**：
           - 允许用户选择特定的基因集进行建模
           - 适用于已有先验知识的情况，如已知的通路基因
           - 用户可以选择不同的机器学习算法和参数
        
        2. **自动化建模 (Auto Mode)**：
           - 自动执行特征选择和模型构建的完整流程
           - 包括单变量筛选、LASSO特征选择和随机森林优化
           - 适用于探索性分析，寻找最优的基因集
        
        #### 特征选择方法
        
        1. **单变量筛选**：
           - 使用统计测试（如F检验）评估每个特征与目标变量的相关性
           - 保留p值小于阈值的特征
           - 优点：简单、快速；缺点：忽略特征间的相互作用
        
        2. **LASSO特征选择**：
           - 使用L1正则化的线性模型，可以将不重要特征的系数压缩为零
           - 自动执行特征选择，同时考虑特征间的相互作用
           - 优点：能处理高维数据，减少过拟合；缺点：对特征缩放敏感
        
        3. **随机森林特征重要性**：
           - 基于特征在决策树中的平均不纯度减少来评估重要性
           - 能够捕捉非线性关系和特征交互
           - 优点：稳健，不受特征缩放影响；缺点：计算成本较高
        
        #### 模型评估指标
        
        1. **准确率**：正确预测的样本比例
        2. **精确率**：在预测为正类的样本中，真正为正类的比例
        3. **召回率**：在所有真正为正类的样本中，被正确预测为正类的比例
        4. **F1分数**：精确率和召回率的调和平均
        5. **ROC曲线和AUC**：评估模型在不同阈值下的性能
        """)

if __name__ == "__main__":
    machine_learning_page()
