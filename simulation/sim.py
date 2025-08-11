import numpy as np
import pandas as pd
import os
from scipy.stats import nbinom
# 设置随机种子，确保每次运行的结果相同
np.random.seed(42)
outdir = './simulation/data'
if not os.path.exists(outdir):
    os.makedirs(outdir)
prefix_pre = 'sim'
sig_gene_num = 20
# 样本数量与基因数量
num_samples = 1000  # 样本数量
num_genes = 100     # 基因数量
prefix = f"{prefix_pre}_{num_samples}_{num_genes}_{sig_gene_num}"
num_confounders = 3  # 混杂因素数量
# 新增参数配置
num_case = num_samples // 2
num_control = num_samples - num_case
log2fc_threshold = 0.01  # 显著基因的log2倍数变化阈值


# 1. 生成真实的生物学信号
# 假设每个基因的表达水平遵循负二项分布
biological_signal = np.random.normal(loc=5.0, scale=3.0, size=(num_samples, num_genes))  # 均值5，标准差1

# 2. 生成3个显式混杂因素 (Batch, Age, Sex)
confounders = np.zeros((num_samples, num_confounders))
# Batch: 10个批次，每个批次有特定的偏移量
confounders[:, 0] = np.random.randint(0, 10, num_samples)  # 批次
# Age: 模拟年龄分布 (20-80岁)
confounders[:, 1] = np.random.normal(loc=50, scale=15, size=num_samples)
confounders[:, 1] = np.clip(confounders[:, 1], 20, 80)  # 限制在20-80岁之间
# Sex: 0=女性, 1=男性
confounders[:, 2] = np.random.binomial(1, 0.5, size=num_samples)  # 约50%男性

# 负二项分布参数设置
base_means = np.exp(np.random.normal(3, 1.5, num_genes))  # 对数正态分布的均值
dispersions = np.random.uniform(0.1, 0.5, num_genes)      # 分散度参数

# 生成生物学信号（负二项分布）
biological_signal = np.zeros((num_samples, num_genes))
for i in range(num_genes):
    mu = base_means[i]
    var = mu + dispersions[i] * mu**2
    p = mu / var
    n = mu * p / (1 - p)
    biological_signal[:, i] = nbinom.rvs(n, p, size=num_samples)

# 增强型非线性函数库
nonlinear_functions = [
    lambda x: np.sin(x * 2),          # 正弦波
    lambda x: np.log1p(np.abs(x)),    # 对数变换
    lambda x: 1 / (1 + np.exp(-x)),   # Sigmoid函数
    lambda x: np.tanh(x * 0.5),       # 双曲正切
    lambda x: np.exp(-0.1*x),         # 指数衰减
    lambda x: np.sqrt(np.abs(x)),     # 平方根
    lambda x: x**2,                   # 二次函数
    lambda x: np.where(x>0, x, 0.5*x) # 分段函数
]

# 生成复杂非线性混杂效应
confounder_effects = np.zeros((num_samples, num_genes))

# 定义三类基因：batch敏感、age相关、sex差异
gene_types = np.random.choice(['batch', 'age', 'sex', 'none'], 
                             size=num_genes, 
                             p=[0.3, 0.3, 0.2, 0.2])

for gene in range(num_genes):
    gene_type = gene_types[gene]
    
    if gene_type == 'batch':
        # Batch效应: 强批次特异性模式
        batch_effect = np.zeros(num_samples)
        for batch in range(10):
            mask = confounders[:, 0] == batch
            # 每个批次有独特的非线性模式
            func = np.random.choice(nonlinear_functions[:4])
            batch_effect[mask] = func(batch) * np.random.uniform(0.8, 1.5)
        
        # 添加批次间交互效应
        interaction_func = np.random.choice(nonlinear_functions[4:])
        batch_effect += interaction_func(confounders[:, 0]) * 0.3
        
        confounder_effects[:, gene] = batch_effect * np.random.uniform(0.8, 1.2)
    
    elif gene_type == 'age':
        # Age效应: 更符合生物学的年龄相关模式
        age = confounders[:, 1]
        
        # 选择与年龄相关的非线性函数
        age_funcs = [
            lambda x: 0.5 * np.sin(0.1 * (x - 40)),  # 中年表达高峰
            lambda x: 0.01 * (x - 50)**2,            # U型或倒U型
            lambda x: -0.02 * np.abs(x - 60),        # 年龄越大表达越低
            lambda x: np.where(x<50, 0.5, -0.5),     # 50岁前后差异
            lambda x: 1 / (1 + np.exp(-0.1*(x-40)))  # Sigmoid年龄依赖
        ]
        
        age_effect = np.random.choice(age_funcs)(age)
        
        # 添加一些噪声
        age_effect += np.random.normal(0, 0.1, num_samples)
        
        confounder_effects[:, gene] = age_effect * np.random.uniform(0.5, 1.0)
    
    elif gene_type == 'sex':
        # Sex效应: 性别差异基因
        sex = confounders[:, 2]
        
        # 基础性别差异
        sex_effect = np.where(sex == 1, 
                             np.random.normal(0.8, 0.2),  # 男性表达水平
                             np.random.normal(-0.5, 0.2)) # 女性表达水平
        
        # 添加年龄依赖的性别差异
        age = confounders[:, 1]
        age_modulation = np.random.choice([
            lambda x: 0.05 * (x - 40),              # 线性年龄调节
            lambda x: 0.001 * (x - 40)**2,          # 二次年龄调节
            lambda x: np.sin(0.1 * x)               # 周期性年龄调节
        ])(age)
        
        sex_effect = sex_effect * (1 + age_modulation)
        
        confounder_effects[:, gene] = sex_effect * np.random.uniform(0.6, 1.0)
    
    else:
        # 随机混杂效应 (较弱)
        selected_funcs = np.random.choice(nonlinear_functions, size=2, replace=False)
        effect = (
            selected_funcs[0](confounders[:, 0]) * 0.3 +   # Batch效应
            selected_funcs[1](confounders[:, 1]) * 0.2     # Age效应
        )
        confounder_effects[:, gene] = effect * np.random.uniform(0.3, 0.6)

# 应用非线性混杂效应（乘法效应）
gene_expression = biological_signal * (1 + confounder_effects)

# 添加差异表达（在负二项分布框架下使用乘法效应）
log2fc_values = np.concatenate([
    np.random.normal(log2fc_threshold, 0.2, size=sig_gene_num//2),
    np.random.normal(-log2fc_threshold, 0.2, size=sig_gene_num//2)
])
sig_gene_indices = np.random.choice(num_genes, size=sig_gene_num, replace=False)
np.random.shuffle(sig_gene_indices)
for i, gene_idx in enumerate(sig_gene_indices):
    fold_change = 2 ** log2fc_values[i]
    # 处理组样本应用倍数变化
    gene_expression[num_control:, gene_idx] = np.round(
        gene_expression[num_control:, gene_idx] * fold_change
    )

# 生成显著基因元数据（包含log2FC和p值）
sig_genes_metadata = pd.DataFrame({
    'gene_id': [f'gene{j+1}' for j in sig_gene_indices],
    'log2fc': log2fc_values,
    'gene_type': [gene_types[j] for j in sig_gene_indices]  # 添加基因类型信息
}).sort_values('log2fc', ascending=False)

# 保存显著基因列表
sig_genes_metadata.to_csv(
    os.path.join(outdir, f'{prefix}_significant_genes.txt'),
    index=False,
    sep='\t',
    columns=['gene_id', 'log2fc', 'gene_type']
)

# 4. 将生成的数据保存为CSV文件
# 基因表达数据（文件1）
gene_expression_df = pd.DataFrame(gene_expression.astype(int), columns=[f'gene{i+1}' for i in range(num_genes)])
# set value small than 0 to 0
gene_expression_df = gene_expression_df.applymap(lambda x: 0 if x < 0 else x)
gene_expression_df.insert(0, 'Sample', [f'sample{i+1}' for i in range(num_samples)])
gene_expression_df.set_index('Sample', inplace=True, drop=True)
gene_expression_df = gene_expression_df.T

# 显式混杂因素（文件2）
group_labels = np.array(['control']*num_control + ['case']*num_case)
confounders_df = pd.DataFrame(confounders, columns=[f'confounder{i+1}' for i in range(num_confounders)])
confounders_df.insert(0, 'Sample', [f'sample{i+1}' for i in range(num_samples)])
confounders_df.columns = ['Sample'] + ['Batch', 'Age', 'Sex']
confounders_df['Tissue'] = group_labels
# 保存数据为CSV文件
gene_expression_df.to_csv(os.path.join(outdir, f'{prefix}_all_gene_expression.txt'), index=True, sep='\t', index_label='ID')  # 保存到指定路径
confounders_df.to_csv(os.path.join(outdir, f'{prefix}_all_confounders.txt'), index=False, sep='\t')
print("生成的文件已保存。")
