import torch
from models.discriminator import Discriminator, adversarial_loss,TissueDiscriminator, TissueDiscriminator2
from models.ae import TwinAE
from utils.losses import reconstruction_loss
from utils.preprocessing import pca_reduce, kmeans_reduce_genes
import torch.nn.functional as F
from utils.data_loader import GeneExpressionDataset
from torch.utils.data import DataLoader
from torch import nn
from torch.utils.data import WeightedRandomSampler
from torch.optim import Adam
import numpy as np
from torch.nn.modules import Module
import os 
import pandas as pd
from copy import deepcopy
import copy
import argparse
import os
import logging
class EarlyStopping:
    def __init__(self, patience=5, min_delta=0):
        """
        :param patience: 允许的非改善轮数
        :param min_delta: 最小改进量
        """
        self.patience = patience
        self.min_delta = min_delta
        self.best_loss = np.inf
        self.counter = 0
        self.early_stop = False

    def __call__(self, val_loss):
        if val_loss < self.best_loss - self.min_delta:
            self.best_loss = val_loss
            self.counter = 0  # 重置计数器
        else:
            self.counter += 1  # 增加计数器

        if self.counter >= self.patience:
            self.early_stop = True  # 激活早停


class TwinAETrainer:
    def __init__(
        self,
        twin_ae: Module,
        discriminator: Discriminator,
        tissueDiscriminator: TissueDiscriminator,
        # tissueDiscriminator2: TissueDiscriminator2,
        device: torch.device,
        # 训练阶段参数
        stage1_epochs: int = 300,
        stage2_epochs: int = 300,
        stage3_epochs: int = 300,
        stage1_lr: float = 1e-3,
        stage2_lr: float = 1e-3,
        stage3_lr: float = 1e-3,
        lambda_adv: float = 1.0,
        tissue_adv: float = 1.0,
        patience: int = 300,
        verbose: bool = True
    ):
        self.twin_ae = twin_ae.to(device)
        self.discriminator = discriminator.to(device)
        self.tissue_disc = tissueDiscriminator.to(device)
        self.device = device
        
        # 训练参数
        self.stage1_epochs = stage1_epochs
        self.stage2_epochs = stage2_epochs
        self.stage3_epochs = stage3_epochs
        self.stage1_lr = stage1_lr
        self.stage2_lr = stage2_lr
        self.stage3_lr = stage3_lr
        self.lambda_adv = lambda_adv
        self.tissue_adv = tissue_adv
        self.patience = patience
        self.verbose = verbose
        

    def train_stage1(self, train_loader):
        """第一阶段：仅训练自编码器重构"""
        self.twin_ae.train()
        optimizer = Adam(params=list(self.twin_ae.ae.encoder.parameters())+ list(self.twin_ae.ae.decoder.parameters()), lr=self.stage1_lr)
        
        for epoch in range(self.stage1_epochs):
            total_loss = 0.0
            for x1,_ in train_loader:
                x1 = x1.to(self.device)
                
                optimizer.zero_grad()
                x1_hat, _, _ = self.twin_ae(x1)
                loss = reconstruction_loss(x1, x1_hat)
                
                loss.backward()
                optimizer.step()
                
                total_loss += loss.item()
            
            avg_loss = total_loss / len(train_loader)

            if self.verbose and epoch % 10 == 0:
                print(f"Stage1 Epoch [{epoch}/{self.stage1_epochs}] Loss: {avg_loss:.4f}")

    def train_stage2(self, train_loader):
        """第二阶段：加入对抗训练"""
        self.twin_ae.train()
        self.discriminator.train()
        
        twin_optim = Adam(self.twin_ae.parameters(), lr=self.stage2_lr)
        disc_optim = Adam(self.discriminator.parameters(), lr=self.stage2_lr/10)
        best_model_state = None
        best_score = np.inf
        for epoch in range(self.stage2_epochs):
            total_recon = 0.0
            total_adv = 0.0
            total_mi = 0.0
            # self.lambda_adv = min(self.lambda_adv * 1.0001, 1.2)
            for x1,label in train_loader:
                x1 = x1.to(self.device)
                x1_hat, bio1, conf1 = self.twin_ae(x1)
                twin_optim.zero_grad()
                recon_loss = reconstruction_loss(x1, x1_hat)
                _, g_loss = adversarial_loss(bio1, conf1, self.discriminator)
                total_loss = recon_loss + g_loss*self.lambda_adv
                total_loss.backward()
                twin_optim.step()

                # 更新判别器
                disc_optim.zero_grad()
                disc_loss, _ = adversarial_loss(bio1.detach(), conf1.detach(),
                                                self.discriminator)
                disc_loss.backward()
                disc_optim.step()
                # 更新生成器

                # 记录损失
                total_recon += recon_loss.item()
                total_adv += disc_loss.item()
                total_mi += g_loss.item()
                # total_uni += uni_loss.item()
            
            avg_recon = total_recon / len(train_loader)
            avg_adv = total_adv / len(train_loader)
            avg_mi = total_mi / len(train_loader)
            if self.verbose and epoch % 10 == 0:
                print(f"Stage2 Epoch [{epoch}/{self.stage2_epochs}] "
                        f"Recon: {avg_recon:.4f} Adv: {avg_adv:.6f}" 
                        f" MI: {avg_mi:.4f}" 
                        f" Lambda: {self.lambda_adv:.4f}")
            if avg_recon+avg_mi < best_score:
                best_score = avg_recon+avg_mi
                best_model_state = deepcopy(self.twin_ae.state_dict())
        if best_model_state is not None:
            self.twin_ae.load_state_dict(best_model_state)

    def train_stage3(self, train_loader, val_loader=None):
        self.twin_ae.train()
        self.discriminator.train()
        self.tissue_disc.train()
        # self.tissue_disc2.train()
        twin_optim = Adam(self.twin_ae.parameters(), lr=self.stage3_lr)
        disc_optim = Adam(self.discriminator.parameters(), lr=self.stage3_lr/10)
        tissue_disc_optim = Adam(self.tissue_disc.parameters(), lr=self.stage3_lr/10)
        best_model_state = None
        best_score = np.inf
        all_tissue_num = len(np.unique(train_loader.dataset.tissues))
        if val_loader is not None:
            early_stopping = EarlyStopping(patience=self.patience)
        for epoch in range(self.stage3_epochs):
            total_recon = 0.0
            total_adv = 0.0
            total_mi = 0.0
            total_tissue_adv = 0.0
            # self.lambda_adv = min(self.lambda_adv * 1.00001, 0.3)
            for x1, tissue_labels in train_loader:
                x1 = x1.to(self.device)
                tissue_num = len(tissue_labels.unique())
                if tissue_num != all_tissue_num:
                    print(f"Warning: Tissue number {tissue_num} does not match expected {all_tissue_num}. Skipping batch.")
                    continue
                tissue_labels = tissue_labels.squeeze().to(self.device)
                class_counts = torch.bincount(tissue_labels, minlength=tissue_num)
                class_counts = class_counts.float() + 1e-6  # 平滑处理避免除零
                class_weights = 1.0 / class_counts
                class_weights = class_weights / class_weights.sum()
                # 更新生成器
                for params in self.discriminator.parameters():
                    params.requires_grad = False
                for params in self.tissue_disc.parameters():
                    params.requires_grad = False
                
                x1_hat, bio1, conf1 = self.twin_ae(x1)
                recon_loss = reconstruction_loss(x1, x1_hat)
                _, g_loss = adversarial_loss(bio1, conf1, self.discriminator)
                tissue_adv = self.tissue_disc(bio1)
                tissue_loss = F.cross_entropy(tissue_adv, tissue_labels, weight=class_weights)
                total_loss = recon_loss + g_loss*self.lambda_adv + self.tissue_adv*tissue_loss
                twin_optim.zero_grad()
                total_loss.backward()
                twin_optim.step()

                for param in self.discriminator.parameters():
                    param.requires_grad = True
                for param in self.tissue_disc.parameters():
                    param.requires_grad = True
                # 更新判别器
                disc_optim.zero_grad()
                disc_loss, _ = adversarial_loss(bio1.detach(), conf1.detach(),
                                                  self.discriminator)
                disc_loss.backward()
                disc_optim.step()

                # 更新组织判别器
                tissue_disc_optim.zero_grad()
                tissue_disc_loss = F.cross_entropy(self.tissue_disc(bio1.detach()), tissue_labels,
                                                   weight=class_weights)
                tissue_disc_loss.backward()
                tissue_disc_optim.step()
                
                # 记录损失
                total_recon += recon_loss.item()
                total_adv += disc_loss.item()
                total_mi += g_loss.item()
                total_tissue_adv += tissue_disc_loss.item()
            
            avg_recon = total_recon / len(train_loader)
            avg_adv = total_adv / len(train_loader)
            avg_mi = total_mi / len(train_loader)
            avg_tissue_adv = total_tissue_adv / len(train_loader)
            if val_loader is not None:
                avg_recon_val, avg_mi_val, avg_tissue_adv_val = self.validate(val_loader)
                best_score_current = avg_recon_val + 0.5 * avg_mi_val
                if best_score_current < best_score:
                    best_score = best_score_current
                    best_model_state = copy.deepcopy(self.twin_ae.state_dict())
                if self.verbose and epoch % 10 == 0:
                    print(f"Stage3 Epoch [{epoch}/{self.stage3_epochs}] "
                          f"Recon: {avg_recon:.4f} " 
                          f"TissueAdv: {avg_tissue_adv:.4f}"
                          f" MI: {avg_mi:.4f}" 
                          f" Lambda: {self.lambda_adv:.4f}", end=" ")
                    print(f"Validation Recon: {avg_recon_val:.4f} "
                          f"TissueAdv: {avg_tissue_adv_val:.4f}"
                          f" MI: {avg_mi_val:.4f} "
                          f"ALL: {best_score_current:.4f}")
                early_stopping(best_score_current)
                if early_stopping.early_stop:
                    print("Early stopping triggered!")
                    break
            else:
                if self.verbose and epoch % 10 == 0:
                    print(f"Stage3 Epoch [{epoch}/{self.stage3_epochs}] "
                        f"Recon: {avg_recon:.4f} Adv: {avg_adv:.4f} " 
                        f"TissueAdv: {avg_tissue_adv:.4f}"
                        f" MI: {avg_mi:.4f}")

        if best_model_state is not None:
            self.twin_ae.load_state_dict(best_model_state)

    def validate(self, val_loader):
        self.twin_ae.eval()
        self.discriminator.eval()
        self.tissue_disc.eval()
        total_recon = 0.0
        total_mi = 0.0
        total_tissue_adv = 0.0
        with torch.no_grad():
            for x1, tissue_labels in val_loader:
                x1 = x1.to(self.device)
                tissue_num = len(tissue_labels.unique())
                tissue_labels = tissue_labels.squeeze().to(self.device)
                class_counts = torch.bincount(tissue_labels, minlength=tissue_num)
                class_counts = class_counts.float() + 1e-6  # 平滑处理避免除零
                class_weights = 1.0 / class_counts
                class_weights = class_weights / class_weights.sum()
                x1_hat, bio1, conf1 = self.twin_ae(x1)
                recon_loss = reconstruction_loss(x1, x1_hat)
                _, mi_loss = adversarial_loss(bio1, conf1, self.discriminator)
                tissue_adv = self.tissue_disc(bio1)
                tissue_loss = F.cross_entropy(tissue_adv, tissue_labels, weight=class_weights)
                total_recon += recon_loss.item()
                total_mi += mi_loss.item()
                total_tissue_adv += tissue_loss.item()
            avg_recon = total_recon / len(val_loader)
            avg_mi = total_mi / len(val_loader)
            avg_tissue_adv = total_tissue_adv / len(val_loader)
        self.twin_ae.train()
        self.discriminator.train()
        self.tissue_disc.train()
        return avg_recon, avg_mi, avg_tissue_adv

    def train(self, train_loader, val_loader=None):
        """执行完整训练流程"""
        self.train_stage1(train_loader)
        self.train_stage2(train_loader)
        self.train_stage3(train_loader, val_loader=val_loader)
        
    def get_clean_representation(self, dataset):
        """获取去混杂后的表示"""
        loader = DataLoader(dataset, batch_size=len(dataset), shuffle=False)
        self.twin_ae.eval()
        
        clean_embeddings = []
        clean_embeddings3 = []
        bio_ali = []
        conf_ali = []
        with torch.no_grad():
            for x1,_ in loader:
                x1 = x1.to(self.device)
                recon, bio, conf = self.twin_ae(x1)
                clean_embeddings.append(recon.cpu())
                clean_embeddings3.append(self.twin_ae.ae.decoder(torch.cat((bio, conf.mean(dim=0).repeat(x1.shape[0], 1)), dim=1)).cpu())
                bio_ali.append(bio.cpu())
                conf_ali.append(conf.cpu())
        return torch.cat(clean_embeddings, dim=0).numpy(), torch.cat(clean_embeddings3, dim=0).numpy(), torch.cat(bio_ali, dim=0).numpy(), torch.cat(conf_ali, dim=0).numpy()
# 计算每个样本的权重
def get_sample_weights(tissues):
    class_counts = np.bincount(tissues)
    class_weights = 1. / class_counts
    sample_weights = class_weights[tissues]
    return sample_weights

def run_usadae(latent_dim=32, confounder_dim=2, drop_out=0.2, lambda_adv=1.0,lambda_tissue=1e6,
         stage1_epochs=30, stage2_epochs=20,stage3_epochs=20, batch_size=64,hidden_dims=[512, 256],
         out_prefix='usadae',random_seed=0,
         gene_expresion_df=None,tissue_labels=None, tissue_dis_hidden_num=0 ,outdir=None,separate_data=False):
    # set random seed for torch and numpy
    torch.manual_seed(random_seed)
    np.random.seed(random_seed)
    torch.cuda.manual_seed(random_seed)  # 如果使用 GPU，设置 CUDA 随机种子
    input_dim = gene_expresion_df.shape[1] 
    twin_ae = TwinAE(input_dim=input_dim, latent_dim=latent_dim, confounder_dim=confounder_dim, drop_out=drop_out,
                     hidden_dims=hidden_dims)
    discriminator = Discriminator(dim_b=latent_dim, dim_c=confounder_dim)
    tissue_num = len(np.unique(tissue_labels))
    tissue_disc = TissueDiscriminator(latent_dim=latent_dim, output_dim=tissue_num, hidden_num=tissue_dis_hidden_num)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # 创建训练器
    trainer = TwinAETrainer(
        twin_ae=twin_ae,
        discriminator=discriminator,
        tissueDiscriminator=tissue_disc,
        device=device,
        stage1_epochs=stage1_epochs,
        stage2_epochs=stage2_epochs,
        stage3_epochs=stage3_epochs,
        stage1_lr=1e-3,
        stage2_lr=1e-3,  # 可调整不同阶段的学习率
        stage3_lr=1e-3,
        lambda_adv=lambda_adv,    # 对抗损失权重
        tissue_adv=lambda_tissue,  # 病理损失权重
    )
    if separate_data:
        # 随机采样 80% 的索引（基于 gene_expresion_df 的行数）
        n_samples = len(gene_expresion_df)
        train_indices = np.random.choice(n_samples, size=int(0.8 * n_samples), replace=False)

        # 生成一个布尔掩码，标记训练样本
        train_mask = np.zeros(n_samples, dtype=bool)
        train_mask[train_indices] = True

        # 提取训练集和验证集
        gene_expresion_df_train = gene_expresion_df.iloc[train_indices]
        gene_expresion_df_val = gene_expresion_df.drop(gene_expresion_df.index[train_indices])
        # 提取对应的 tissue_labels
        tissue_labels_train = tissue_labels[train_mask]
        tissue_labels_val = tissue_labels[~train_mask]
        # 创建数据集
        train_dataset = GeneExpressionDataset(gene_expresion_df_train, tissues=tissue_labels_train)
        val_dataset = GeneExpressionDataset(gene_expresion_df_val, tissues=tissue_labels_val)
        sample_weights = get_sample_weights(train_dataset.tissues)
        sampler = WeightedRandomSampler(sample_weights, len(sample_weights), replacement=True)
        # 创建 DataLoader
        train_loader = DataLoader(train_dataset, batch_size=batch_size, sampler=sampler)
        val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)  # 验证集通常不 shuffle

        # 训练模型
        trainer.train(train_loader, val_loader=val_loader)
    else:

        all_dataset = GeneExpressionDataset(gene_expresion_df, tissues=tissue_labels)
        # 在创建 DataLoader 时使用
        sample_weights = get_sample_weights(all_dataset.tissues)
        sampler = WeightedRandomSampler(sample_weights, len(sample_weights), replacement=True)
        train_loader = DataLoader(all_dataset, batch_size=batch_size, sampler=sampler)
        trainer.train(train_loader)

    all_dataset = GeneExpressionDataset(gene_expresion_df, tissues=tissue_labels)
    # 获取去混杂后的数据
    clean_emb, clean_emb3, bio, conf = trainer.get_clean_representation(all_dataset)

    df = pd.DataFrame(clean_emb, index=all_dataset.gene_expression_data.index, columns=all_dataset.gene_expression_data.columns).T
    df.to_csv(os.path.join(outdir, f'{out_prefix}_ori_recon.txt'), sep='\t', index=True, index_label='ID')

    df3 = pd.DataFrame(clean_emb3, index=all_dataset.gene_expression_data.index, columns=all_dataset.gene_expression_data.columns).T
    df3.to_csv(os.path.join(outdir, f'{out_prefix}_corrected_recon.txt'), sep='\t', index=True, index_label='ID')

    df2 = pd.DataFrame(bio, index=all_dataset.gene_expression_data.index, columns=[f'Bio_{i}' for i in range(bio.shape[1])]).T
    df2.to_csv(os.path.join(outdir, f'{out_prefix}_bio_latent.txt'), sep='\t', index=True, index_label='ID')

    df4 = pd.DataFrame(conf, index=all_dataset.gene_expression_data.index, columns=[f'Conf_{i}' for i in range(conf.shape[1])]).T
    df4.to_csv(os.path.join(outdir, f'{out_prefix}_conf_latent.txt'), sep='\t', index=True, index_label='ID')

def main():
    parser = argparse.ArgumentParser(description="USADAE: Deep learning for separating confounding factors in RNA-seq data.")
    
    # 核心参数
    parser.add_argument("--latent_dim", type=int, default=12, help="Latent dimension (default: %(default)s)")
    parser.add_argument("--confounder_dim", type=int, default=None, help="Confounder dimension (default: same as latent_dim)")
    parser.add_argument("--dropout", type=float, default=0.4, help="Dropout rate (default: %(default)s)")
    parser.add_argument("--lambda_adv", type=float, default=1.0, help="Adversarial loss weight (default: %(default)s)")
    parser.add_argument("--lambda_tissue", type=float, default=10, help="Tissue loss weight (default: %(default)s)")
    parser.add_argument("--hidden_dims", nargs="*", type=int, default=[512, 256], help="Hidden layer dimensions (e.g., 512 256; default: %(default)s)")

    # 训练参数
    parser.add_argument("--stage1_epochs", type=int, default=100, help="Stage 1 epochs (default: %(default)s)")
    parser.add_argument("--stage2_epochs", type=int, default=100, help="Stage 2 epochs (default: %(default)s)")
    parser.add_argument("--stage3_epochs", type=int, default=500, help="Stage 3 epochs (default: %(default)s)")
    parser.add_argument("--batch_size", type=int, default=128, help="Batch size (default: %(default)s)")
    parser.add_argument("--seed", type=int, default=0, help="Random seed (default: %(default)s)")

    # 文件和输出
    parser.add_argument("--gene_expression_file", type=str, required=True, help="Path to gene expression TSV/CSV file. Rows are genes, columns are samples.")
    parser.add_argument("--confounder_file", type=str, required=True, help="Path to confounders TSV/CSV file (with 'Tissue' column). Rows are samples.")
    parser.add_argument("--outdir", type=str, required=True, help="Output directory")
    parser.add_argument("--out_prefix", type=str, default="usadae", help="Output file prefix (default: %(default)s)")
    
    # 预处理和数据分割
    parser.add_argument("--log1p", action="store_true", help="Apply log1p transformation (default: False)")
    parser.add_argument("--separate_data", action="store_true", help="Separate data into train/val (80/20 split; default: False)")
    parser.add_argument("--use_kmeans", action="store_true", help="Apply KMeans gene clustering before training (default: False)")
    parser.add_argument("--kmeans_clusters", type=int, default=1500, help="Number of KMeans clusters if --use_kmeans (default: %(default)s)")
    args = parser.parse_args()
    # 设置日志
    logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
    logger = logging.getLogger(__name__)
    try:
        logger.info("Loading gene expression data from %s", args.gene_expression_file)
        if args.gene_expression_file.endswith('.csv'):
            exp_df = pd.read_csv(args.gene_expression_file, index_col=0).T
        elif args.gene_expression_file.endswith('.tsv') or args.gene_expression_file.endswith('.txt'):
            exp_df = pd.read_csv(args.gene_expression_file, sep='\t', index_col=0).T
        else:
            raise ValueError("Unsupported file format. Please provide a CSV or TSV file.")
        if args.log1p:
            logger.info("Applying log1p transformation to gene expression data")
            exp_df = exp_df.apply(np.log1p)
        if args.use_kmeans:
            logger.info("Applying KMeans clustering (n_clusters: %d)", args.kmeans_clusters)
            exp_df = kmeans_reduce_genes(exp_df.apply(np.log1p), n_clusters=args.kmeans_clusters)
        exp_df = (exp_df - exp_df.mean()) / exp_df.std()  # 标准化
        logger.info("Gene expression data shape: %s", exp_df.shape)
        logger.info("Loading confounder data from %s", args.confounder_file)
        if args.confounder_file.endswith('.csv'):
            confounder_df = pd.read_csv(args.confounder_file, index_col=0)
        elif args.confounder_file.endswith('.tsv') or args.confounder_file.endswith('.txt'):
            confounder_df = pd.read_csv(args.confounder_file, sep='\t', index_col=0)
        else:
            raise ValueError("Unsupported confounder file format. Please provide a CSV or TSV file.")
        if 'Tissue' not in confounder_df.columns:
            raise ValueError("Confounder file must contain a 'Tissue' column.")
        logger.info("Confounder data shape: %s", confounder_df.shape)
        labels = confounder_df['Tissue'].values
    except Exception as e:
        logger.error("Data loading failed: %s", e)
        raise
    confounder_dim = args.confounder_dim if args.confounder_dim is not None else args.latent_dim
    if not os.path.exists(args.outdir):
        os.makedirs(args.outdir)
    run_usadae(
        latent_dim=args.latent_dim,
        confounder_dim=confounder_dim,
        drop_out=args.dropout,
        lambda_adv=args.lambda_adv,
        lambda_tissue=args.lambda_tissue,
        hidden_dims=args.hidden_dims,
        stage1_epochs=args.stage1_epochs,
        stage2_epochs=args.stage2_epochs,
        stage3_epochs=args.stage3_epochs,
        batch_size=args.batch_size,
        out_prefix=args.out_prefix,
        outdir=args.outdir,
        gene_expresion_df=exp_df,
        random_seed=args.seed,
        tissue_labels=labels,
        separate_data=args.separate_data
    )

if __name__ == "__main__":
    # import pandas as pd
    # import numpy as np
    # # 初始化模型
    # latent_dims = [16]
    # drop_outs = [0.4]
    # # confouder_dims = [6]
    # lambda_advs = [1]
    # hidden_dimss = [[512,256]]
    # gene_expresion_file = '/home/chenxu/work/aself_requirement/20250216.deepeer/20250321.simulate_data/analysis4.2/sim_10000_1000_200_all_gene_expression.txt'
    # confounder_dim = '/home/chenxu/work/aself_requirement/20250216.deepeer/20250321.simulate_data/analysis4.2/sim_10000_1000_200_all_confounders.txt'
    
    # exp_df = pd.read_csv(gene_expresion_file, sep='\t', index_col=0).T
    # # exp_df = pca_reduce(exp_df)
    # # exp_df = kmeans_reduce_genes(exp_df.apply(np.log1p), n_clusters=1000)
    # exp_df = (exp_df - exp_df.mean()) / exp_df.std()
    # print(exp_df.shape)
    # confouder_df = pd.read_csv(confounder_dim, sep='\t', index_col=0)
    # labels = confouder_df['Tissue'].values
    # outdir  = '/home/chenxu/work/aself_requirement/20250216.deepeer/deepeer_1.1_tune2_add_dis3/test_sim4.2_withval'
    # if not os.path.exists(outdir):
    #     os.makedirs(outdir)
    # for latent_dim in latent_dims:
    #     for drop_out in drop_outs:
    #         confouder_dim = latent_dim
    #         for lambda_adv in lambda_advs:
    #             print(latent_dim, confouder_dim, drop_out, lambda_adv)
    #             run_usadae(latent_dim=latent_dim, confouder_dim=confouder_dim, drop_out=drop_out, lambda_adv=lambda_adv,
    #                 gene_expresion_df=exp_df, outdir=outdir, tissue_labels=labels,hidden_dims=hidden_dimss[0],
    #                 stage1_epochs=100, stage2_epochs=100,stage3_epochs=600,batch_size=500, separate_data=False
    #             )
    main()