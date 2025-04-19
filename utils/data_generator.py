import pandas as pd
import numpy as np
import random


def generate_gene_expression(n_samples, n_genes):
    """
    生成基因表达数据

    参数:
    n_samples: 样本数量
    n_genes: 基因数量

    返回:
    numpy.ndarray: 基因表达数据
    """
    # RNA-seq数据通常呈现对数正态分布
    gene_data = np.exp(np.random.normal(0, 1, size=(n_samples, n_genes)))
    return gene_data


def generate_clinical_features(n_samples, n_genes, n_clinical):
    """
    生成临床特征数据

    参数:
    n_samples: 样本数量
    n_genes: 基因数量
    n_clinical: 临床特征数量

    返回:
    dict: 包含临床特征的字典
    """
    clinical_data = {}
    # 年龄 (连续变量)
    clinical_data['age'] = np.random.normal(60, 10, n_samples)
    # 性别 (二分类变量)
    clinical_data['gender'] = np.random.choice(['Male', 'Female'], n_samples)

    # 疾病状态 (二分类变量，作为目标变量)
    # 使疾病状态与某些基因表达相关
    # 选择10个随机基因作为"信号"基因
    signal_genes = np.random.choice(n_genes, 10, replace=False)
    gene_data = generate_gene_expression(n_samples, n_genes)
    # 基于这些基因的表达创建疾病状态
    # 如果信号基因的平均表达量高于中位数，则更可能是疾病状态
    signal = np.mean(gene_data[:, signal_genes], axis=1)
    median_signal = np.median(signal)
    # 添加一些随机性，使其不是完全确定的
    prob_disease = 1 / (1 + np.exp(-(signal - median_signal)))
    clinical_data['disease_status'] = np.random.binomial(1, prob_disease)
    clinical_data['disease_status'] = ['Diseased' if x == 1 else 'Healthy' for x in clinical_data['disease_status']]

    # 生成其他临床特征
    for i in range(3, n_clinical + 1):
        if i % 3 == 0:  # 每三个特征添加一个分类变量
            clinical_data[f'clinical_{i}'] = np.random.choice(['Low', 'Medium', 'High'], n_samples)
        else:  # 其他为连续变量
            clinical_data[f'clinical_{i}'] = np.random.normal(50, 15, n_samples)

    return clinical_data


def generate_rna_seq_data(n_samples=100, n_genes=200, n_clinical=5):
    """
    生成RNA-seq示例数据

    参数:
    n_samples: 样本数量
    n_genes: 基因数量
    n_clinical: 临床特征数量

    返回:
    pandas DataFrame: 包含RNA-seq表达数据和临床特征的数据框
    """
    # 检查输入参数是否有效
    if n_samples <= 0 or n_genes <= 0 or n_clinical <= 0:
        raise ValueError("样本数量、基因数量和临床特征数量必须为正整数。")

    # 设置随机种子以确保可重复性
    np.random.seed(42)
    random.seed(42)

    # 生成基因表达数据
    gene_data = generate_gene_expression(n_samples, n_genes)
    # 创建基因名称
    gene_names = [f"GENE_{i + 1}" for i in range(n_genes)]
    # 创建样本ID
    sample_ids = [f"SAMPLE_{i + 1}" for i in range(n_samples)]
    # 创建基因表达数据框
    gene_df = pd.DataFrame(gene_data, columns=gene_names, index=sample_ids)

    # 生成临床特征
    clinical_data = generate_clinical_features(n_samples, n_genes, n_clinical)
    # 创建临床数据框
    clinical_df = pd.DataFrame(clinical_data, index=sample_ids)

    # 合并基因表达和临床数据
    combined_df = pd.concat([clinical_df, gene_df], axis=1)
    # 重置索引，将样本ID作为列
    combined_df = combined_df.reset_index().rename(columns={'index': 'sample_id'})

    return combined_df