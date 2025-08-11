import pandas as pd
from sklearn.decomposition import PCA
from sklearn.cluster import KMeans
def pca_reduce(data:pd.DataFrame, variance_threshold=0.95):
    pca = PCA(n_components=variance_threshold)
    pca.fit(data)
    reduced_data = pca.transform(data)
    return pd.DataFrame(reduced_data, index=data.index, columns=[f'PC{i+1}' for i in range(reduced_data.shape[1])])

def kmeans_reduce_genes(
    data: pd.DataFrame, 
    n_clusters: int = 1000, 
    random_state: int = 42
) -> pd.DataFrame:
    """
    Reduce gene expression data by clustering genes using k-means++.
    Returns a new DataFrame where each sample is represented by gene cluster centroids.
    
    Args:
        data (pd.DataFrame): Preprocessed gene expression data (samples × genes).
        n_clusters (int): Number of gene clusters (default=1000).
        random_state (int): Random seed for reproducibility.
    
    Returns:
        pd.DataFrame: Shape (n_samples × n_clusters), columns are gene cluster centroids.
    """
    # Step 1: Transpose to cluster genes (genes × samples)
    gene_data = data.T  # Shape: (n_genes, n_samples)
    
    # Step 2: Apply k-means++ to cluster genes
    kmeans = KMeans(
        n_clusters=n_clusters,
        n_init='auto',
        init="k-means++",
        random_state=random_state
    ).fit(gene_data)
    
    # Step 3: Extract cluster centers and transpose
    # cluster_centers_ shape: (n_clusters, n_samples)
    # Transpose to get (n_samples × n_clusters)
    cluster_centers = kmeans.cluster_centers_.T
    
    # Step 4: Create DataFrame with cluster features
    reduced_df = pd.DataFrame(
        cluster_centers,
        index=data.index,  # Preserve sample IDs
        columns=[f"GeneCluster_{i+1}" for i in range(n_clusters)]
    )
    
    return reduced_df
