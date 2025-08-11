import pandas as pd
import numpy as np
from sklearn.decomposition import PCA
import matplotlib.pyplot as plt
import seaborn as sns
import os
from glob import glob

# 读取基因表达数据

file_paths = glob(r'./simulation/result/**.txt')
confouder_path = f'./simulation/data/sim_1000_100_20_all_confounders.txt'
for file_path in file_paths:
    fig_name = os.path.basename(file_path).replace(".txt", "")
    outdir = os.path.dirname(file_path)
    # if os.path.exists(os.path.join(outdir, f"{fig_name}.png")):
    #     continue
    confouder = pd.read_csv(confouder_path, sep='\t', index_col=0)
    data = pd.read_csv(file_path, sep='\t', index_col=0).T
    # data = data.apply(np.log1p)  # 对数转换
    common_sample = data.index.intersection(confouder.index)
    confouder = confouder.loc[common_sample]
    data = data.loc[common_sample]

    # 进行PCA分析
    pca = PCA(n_components=2)
    pca_result = pca.fit_transform(data)
    pca_df = pd.DataFrame(pca_result, columns=['PC1', 'PC2'])
    pca_df['Sex'] = confouder['Sex'].astype(str).to_list()
    pca_df['Batch'] = confouder["Batch"].astype('category').to_list()  # 确保作为分类变量
    pca_df['Age'] = confouder['Age'].astype(float).to_list()
    pca_df['Tissue'] = confouder['Tissue'].astype('category').to_list()

    # 创建子图和调色板配置
    fig, axes = plt.subplots(figsize=(15, 3), ncols=4, nrows=1)
    palette_config = {
        'Batch': ('tab10', 'legend', 'Batch'),
        'Sex': ('Set1', 'legend', 'Sex'),
        'Age': ('viridis', 'colorbar', 'Age'),
        'Tissue': ('Set2', 'legend', 'Tissue')
    }

    # 统一坐标轴范围
    xmin, xmax = pca_df['PC1'].min(), pca_df['PC1'].max()
    ymin, ymax = pca_df['PC2'].min(), pca_df['PC2'].max()

    for ax, (hue_var, (palette, l_type, title)) in zip(axes, palette_config.items()):
        sns.scatterplot(
            data=pca_df,
            x='PC1',
            y='PC2',
            hue=hue_var,
            palette=palette,
            ax=ax,
            s=40,
            alpha=0.7,
            edgecolor='w',
            linewidth=0.5
        )
        
        # 设置坐标轴范围和标签
        ax.set_xlim(xmin - 0.1*(xmax-xmin), xmax + 0.1*(xmax-xmin))
        ax.set_ylim(ymin - 0.1*(ymax-ymin), ymax + 0.1*(ymax-ymin))
        ax.set_xlabel('PC1' if ax != axes[0] else f'PC1 ({pca.explained_variance_ratio_[0]:.2%})')
        ax.set_ylabel('PC2' if ax != axes[0] else f'PC2 ({pca.explained_variance_ratio_[1]:.2%})')
        
        # 处理图例或颜色条
        ax.set_title(f'By {title}')
        if l_type == 'legend':
            ax.legend(
                bbox_to_anchor=(1.05, 1),
                loc='upper left',
                title=title,
                frameon=False,
                ncol=1 if len(pca_df[hue_var].unique()) < 12 else 2
            )
        elif l_type == 'colorbar':
            norm = plt.Normalize(pca_df[hue_var].min(), pca_df[hue_var].max())
            sm = plt.cm.ScalarMappable(norm=norm, cmap=palette)
            sm.set_array([])
            plt.colorbar(sm, ax=ax, label=title, shrink=0.8)
            ax.get_legend().remove()

    plt.tight_layout()
    output_path = os.path.join(outdir, f'{fig_name}.png')
    plt.savefig(output_path, bbox_inches='tight', dpi=300)
    # plt.savefig(output_path.replace('.png', '.pdf'), bbox_inches='tight', dpi=300)
    plt.close()