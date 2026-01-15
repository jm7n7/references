"""
Unsupervised Learning Models in Machine Learning
This file contains template implementations for various clustering, dimensionality reduction,
and anomaly detection models.
"""

import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import (
    silhouette_score, calinski_harabasz_score, davies_bouldin_score,
    adjusted_rand_score, normalized_mutual_info_score,
    mean_squared_error
)

# ============================================================================
# CLUSTERING MODELS
# ============================================================================

# 1. K-MEANS CLUSTERING
def kmeans_example(X, n_clusters=3, random_state=42):
    """
    K-Means clustering - partitions data into k clusters.
    
    Args:
        X: Feature matrix
        n_clusters: Number of clusters
        random_state: Random seed
    
    Returns:
        model, labels, cluster_centers
    """
    from sklearn.cluster import KMeans
    
    # Scale features (important for K-Means)
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    
    # Initialize and fit
    model = KMeans(
        n_clusters=n_clusters,
        init='k-means++',  # or 'random'
        n_init=10,
        max_iter=300,
        random_state=random_state
    )
    labels = model.fit_predict(X_scaled)
    cluster_centers = model.cluster_centers_
    
    # Evaluation
    silhouette = silhouette_score(X_scaled, labels)
    calinski_harabasz = calinski_harabasz_score(X_scaled, labels)
    davies_bouldin = davies_bouldin_score(X_scaled, labels)
    
    print(f"K-Means (k={n_clusters}) - Silhouette: {silhouette:.4f}")
    print(f"Calinski-Harabasz: {calinski_harabasz:.4f}, Davies-Bouldin: {davies_bouldin:.4f}")
    
    return model, scaler, labels, cluster_centers


# 2. HIERARCHICAL CLUSTERING (AGGLOMERATIVE)
def hierarchical_clustering_example(X, n_clusters=3, linkage='ward'):
    """
    Hierarchical/Agglomerative Clustering.
    
    Args:
        X: Feature matrix
        n_clusters: Number of clusters
        linkage: Linkage criterion ('ward', 'complete', 'average', 'single')
    
    Returns:
        model, labels
    """
    from sklearn.cluster import AgglomerativeClustering
    
    # Scale features
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    
    # Initialize and fit
    model = AgglomerativeClustering(
        n_clusters=n_clusters,
        linkage=linkage
    )
    labels = model.fit_predict(X_scaled)
    
    # Evaluation
    silhouette = silhouette_score(X_scaled, labels)
    print(f"Hierarchical Clustering (k={n_clusters}, linkage={linkage}) - Silhouette: {silhouette:.4f}")
    
    return model, scaler, labels


# 3. DBSCAN (Density-Based Clustering)
def dbscan_example(X, eps=0.5, min_samples=5):
    """
    DBSCAN - Density-Based Spatial Clustering of Applications with Noise.
    Automatically determines number of clusters.
    
    Args:
        X: Feature matrix
        eps: Maximum distance between samples in same neighborhood
        min_samples: Minimum samples in neighborhood for core point
    
    Returns:
        model, labels, n_clusters
    """
    from sklearn.cluster import DBSCAN
    
    # Scale features
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    
    # Initialize and fit
    model = DBSCAN(eps=eps, min_samples=min_samples)
    labels = model.fit_predict(X_scaled)
    
    # Number of clusters (excluding noise points labeled as -1)
    n_clusters = len(set(labels)) - (1 if -1 in labels else 0)
    n_noise = list(labels).count(-1)
    
    print(f"DBSCAN - Number of clusters: {n_clusters}, Noise points: {n_noise}")
    
    if n_clusters > 1:
        # Only calculate silhouette if we have more than 1 cluster
        silhouette = silhouette_score(X_scaled, labels)
        print(f"Silhouette Score: {silhouette:.4f}")
    
    return model, scaler, labels, n_clusters


# 4. MEAN SHIFT CLUSTERING
def mean_shift_example(X, bandwidth=None):
    """
    Mean Shift clustering - finds clusters without specifying number of clusters.
    
    Args:
        X: Feature matrix
        bandwidth: Bandwidth parameter (if None, estimated automatically)
    
    Returns:
        model, labels, n_clusters
    """
    from sklearn.cluster import MeanShift
    
    # Scale features
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    
    # Initialize and fit
    model = MeanShift(bandwidth=bandwidth)
    labels = model.fit_predict(X_scaled)
    n_clusters = len(set(labels))
    
    print(f"Mean Shift - Number of clusters: {n_clusters}")
    
    if n_clusters > 1:
        silhouette = silhouette_score(X_scaled, labels)
        print(f"Silhouette Score: {silhouette:.4f}")
    
    return model, scaler, labels, n_clusters


# 5. AFFINITY PROPAGATION
def affinity_propagation_example(X, damping=0.5, preference=None):
    """
    Affinity Propagation - creates clusters by sending messages between pairs of samples.
    
    Args:
        X: Feature matrix
        damping: Damping factor (0.5 to 1.0)
        preference: Preference for exemplars (if None, uses median of similarities)
    
    Returns:
        model, labels, n_clusters
    """
    from sklearn.cluster import AffinityPropagation
    
    # Scale features
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    
    # Initialize and fit
    model = AffinityPropagation(damping=damping, preference=preference, random_state=42)
    labels = model.fit_predict(X_scaled)
    n_clusters = len(set(labels))
    
    print(f"Affinity Propagation - Number of clusters: {n_clusters}")
    
    if n_clusters > 1:
        silhouette = silhouette_score(X_scaled, labels)
        print(f"Silhouette Score: {silhouette:.4f}")
    
    return model, scaler, labels, n_clusters


# 6. SPECTRAL CLUSTERING
def spectral_clustering_example(X, n_clusters=3, affinity='rbf'):
    """
    Spectral Clustering - uses eigenvalues of similarity matrix.
    
    Args:
        X: Feature matrix
        n_clusters: Number of clusters
        affinity: How to construct affinity matrix ('rbf', 'nearest_neighbors', 'precomputed')
    
    Returns:
        model, labels
    """
    from sklearn.cluster import SpectralClustering
    
    # Scale features
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    
    # Initialize and fit
    model = SpectralClustering(
        n_clusters=n_clusters,
        affinity=affinity,
        random_state=42
    )
    labels = model.fit_predict(X_scaled)
    
    # Evaluation
    silhouette = silhouette_score(X_scaled, labels)
    print(f"Spectral Clustering (k={n_clusters}) - Silhouette: {silhouette:.4f}")
    
    return model, scaler, labels


# 7. GAUSSIAN MIXTURE MODEL (GMM)
def gmm_example(X, n_components=3, covariance_type='full', random_state=42):
    """
    Gaussian Mixture Model - probabilistic clustering using Gaussian distributions.
    
    Args:
        X: Feature matrix
        n_components: Number of mixture components
        covariance_type: Type of covariance ('full', 'tied', 'diag', 'spherical')
        random_state: Random seed
    
    Returns:
        model, labels, probabilities
    """
    from sklearn.mixture import GaussianMixture
    
    # Scale features
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    
    # Initialize and fit
    model = GaussianMixture(
        n_components=n_components,
        covariance_type=covariance_type,
        random_state=random_state
    )
    model.fit(X_scaled)
    
    # Predictions
    labels = model.predict(X_scaled)
    probabilities = model.predict_proba(X_scaled)
    
    # Evaluation
    silhouette = silhouette_score(X_scaled, labels)
    aic = model.aic(X_scaled)
    bic = model.bic(X_scaled)
    
    print(f"GMM (components={n_components}) - Silhouette: {silhouette:.4f}")
    print(f"AIC: {aic:.4f}, BIC: {bic:.4f}")
    
    return model, scaler, labels, probabilities


# ============================================================================
# DIMENSIONALITY REDUCTION
# ============================================================================

# 8. PRINCIPAL COMPONENT ANALYSIS (PCA)
def pca_example(X, n_components=None, random_state=42):
    """
    Principal Component Analysis - linear dimensionality reduction.
    
    Args:
        X: Feature matrix
        n_components: Number of components (if None, keeps all, or use float for variance ratio)
        random_state: Random seed
    
    Returns:
        model, X_transformed, explained_variance_ratio
    """
    from sklearn.decomposition import PCA
    
    # Scale features (important for PCA)
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    
    # Initialize and fit
    model = PCA(n_components=n_components, random_state=random_state)
    X_transformed = model.fit_transform(X_scaled)
    
    # Explained variance
    explained_variance_ratio = model.explained_variance_ratio_
    cumulative_variance = np.cumsum(explained_variance_ratio)
    
    n_comp = len(explained_variance_ratio)
    print(f"PCA (n_components={n_comp})")
    print(f"Explained variance ratio: {explained_variance_ratio[:5]}...")
    print(f"Cumulative variance (first 5): {cumulative_variance[:5]}...")
    
    return model, scaler, X_transformed, explained_variance_ratio


# 9. INDEPENDENT COMPONENT ANALYSIS (ICA)
def ica_example(X, n_components=None, random_state=42):
    """
    Independent Component Analysis - finds independent sources.
    
    Args:
        X: Feature matrix
        n_components: Number of components
        random_state: Random seed
    
    Returns:
        model, X_transformed
    """
    from sklearn.decomposition import FastICA
    
    # Scale features
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    
    # Initialize and fit
    model = FastICA(n_components=n_components, random_state=random_state, max_iter=1000)
    X_transformed = model.fit_transform(X_scaled)
    
    print(f"ICA (n_components={X_transformed.shape[1]})")
    
    return model, scaler, X_transformed


# 10. FACTOR ANALYSIS
def factor_analysis_example(X, n_components=None, random_state=42):
    """
    Factor Analysis - finds latent factors.
    
    Args:
        X: Feature matrix
        n_components: Number of components
        random_state: Random seed
    
    Returns:
        model, X_transformed
    """
    from sklearn.decomposition import FactorAnalysis
    
    # Scale features
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    
    # Initialize and fit
    model = FactorAnalysis(n_components=n_components, random_state=random_state)
    X_transformed = model.fit_transform(X_scaled)
    
    print(f"Factor Analysis (n_components={X_transformed.shape[1]})")
    
    return model, scaler, X_transformed


# 11. t-SNE (t-Distributed Stochastic Neighbor Embedding)
def tsne_example(X, n_components=2, perplexity=30.0, random_state=42):
    """
    t-SNE - non-linear dimensionality reduction, good for visualization.
    
    Args:
        X: Feature matrix
        n_components: Number of dimensions (typically 2 or 3 for visualization)
        perplexity: Balance between local and global structure (typically 5-50)
        random_state: Random seed
    
    Returns:
        model, X_transformed
    """
    from sklearn.manifold import TSNE
    
    # Scale features
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    
    # Initialize and fit
    model = TSNE(
        n_components=n_components,
        perplexity=perplexity,
        random_state=random_state,
        n_iter=1000
    )
    X_transformed = model.fit_transform(X_scaled)
    
    print(f"t-SNE (n_components={n_components}, perplexity={perplexity})")
    
    return model, scaler, X_transformed


# 12. UMAP (Uniform Manifold Approximation and Projection)
def umap_example(X, n_components=2, n_neighbors=15, min_dist=0.1, random_state=42):
    """
    UMAP - non-linear dimensionality reduction, faster than t-SNE.
    Requires: pip install umap-learn
    
    Args:
        X: Feature matrix
        n_components: Number of dimensions
        n_neighbors: Number of neighbors for local structure
        min_dist: Minimum distance between points in embedding
        random_state: Random seed
    
    Returns:
        model, X_transformed
    """
    try:
        import umap
        
        # Scale features
        scaler = StandardScaler()
        X_scaled = scaler.fit_transform(X)
        
        # Initialize and fit
        model = umap.UMAP(
            n_components=n_components,
            n_neighbors=n_neighbors,
            min_dist=min_dist,
            random_state=random_state
        )
        X_transformed = model.fit_transform(X_scaled)
        
        print(f"UMAP (n_components={n_components}, n_neighbors={n_neighbors})")
        
        return model, scaler, X_transformed
    except ImportError:
        print("UMAP not installed. Install with: pip install umap-learn")
        return None, None, None


# 13. ISOMAP
def isomap_example(X, n_components=2, n_neighbors=5):
    """
    Isomap - Isometric Mapping, non-linear dimensionality reduction.
    
    Args:
        X: Feature matrix
        n_components: Number of dimensions
        n_neighbors: Number of neighbors for graph construction
    
    Returns:
        model, X_transformed
    """
    from sklearn.manifold import Isomap
    
    # Scale features
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    
    # Initialize and fit
    model = Isomap(n_components=n_components, n_neighbors=n_neighbors)
    X_transformed = model.fit_transform(X_scaled)
    
    print(f"Isomap (n_components={n_components}, n_neighbors={n_neighbors})")
    
    return model, scaler, X_transformed


# 14. LOCALLY LINEAR EMBEDDING (LLE)
def lle_example(X, n_components=2, n_neighbors=5, method='standard'):
    """
    Locally Linear Embedding - non-linear dimensionality reduction.
    
    Args:
        X: Feature matrix
        n_components: Number of dimensions
        n_neighbors: Number of neighbors
        method: 'standard', 'modified', 'hessian', 'ltsa'
    
    Returns:
        model, X_transformed
    """
    from sklearn.manifold import LocallyLinearEmbedding
    
    # Scale features
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    
    # Initialize and fit
    model = LocallyLinearEmbedding(
        n_components=n_components,
        n_neighbors=n_neighbors,
        method=method,
        random_state=42
    )
    X_transformed = model.fit_transform(X_scaled)
    
    print(f"LLE (n_components={n_components}, method={method})")
    
    return model, scaler, X_transformed


# ============================================================================
# ANOMALY DETECTION
# ============================================================================

# 15. ISOLATION FOREST
def isolation_forest_example(X, contamination=0.1, random_state=42):
    """
    Isolation Forest - detects anomalies using random forests.
    
    Args:
        X: Feature matrix
        contamination: Expected proportion of anomalies
        random_state: Random seed
    
    Returns:
        model, labels, anomaly_scores
    """
    from sklearn.ensemble import IsolationForest
    
    # Scale features
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    
    # Initialize and fit
    model = IsolationForest(
        contamination=contamination,
        random_state=random_state
    )
    labels = model.fit_predict(X_scaled)  # -1 for anomalies, 1 for normal
    anomaly_scores = model.score_samples(X_scaled)
    
    n_anomalies = list(labels).count(-1)
    print(f"Isolation Forest - Anomalies detected: {n_anomalies} ({n_anomalies/len(labels)*100:.2f}%)")
    
    return model, scaler, labels, anomaly_scores


# 16. LOCAL OUTLIER FACTOR (LOF)
def lof_example(X, n_neighbors=20, contamination=0.1):
    """
    Local Outlier Factor - density-based anomaly detection.
    
    Args:
        X: Feature matrix
        n_neighbors: Number of neighbors
        contamination: Expected proportion of anomalies
    
    Returns:
        model, labels, outlier_scores
    """
    from sklearn.neighbors import LocalOutlierFactor
    
    # Scale features
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    
    # Initialize and fit
    model = LocalOutlierFactor(
        n_neighbors=n_neighbors,
        contamination=contamination
    )
    labels = model.fit_predict(X_scaled)  # -1 for anomalies, 1 for normal
    outlier_scores = model.negative_outlier_factor_
    
    n_anomalies = list(labels).count(-1)
    print(f"LOF (n_neighbors={n_neighbors}) - Anomalies detected: {n_anomalies} ({n_anomalies/len(labels)*100:.2f}%)")
    
    return model, scaler, labels, outlier_scores


# 17. ONE-CLASS SVM
def one_class_svm_example(X, nu=0.1, kernel='rbf', gamma='scale'):
    """
    One-Class SVM - learns a decision boundary for anomaly detection.
    
    Args:
        X: Feature matrix
        nu: Upper bound on fraction of outliers
        kernel: Kernel type ('rbf', 'linear', 'poly', 'sigmoid')
        gamma: Kernel coefficient
    
    Returns:
        model, labels, decision_scores
    """
    from sklearn.svm import OneClassSVM
    
    # Scale features
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    
    # Initialize and fit
    model = OneClassSVM(nu=nu, kernel=kernel, gamma=gamma)
    labels = model.fit_predict(X_scaled)  # -1 for anomalies, 1 for normal
    decision_scores = model.decision_function(X_scaled)
    
    n_anomalies = list(labels).count(-1)
    print(f"One-Class SVM (nu={nu}) - Anomalies detected: {n_anomalies} ({n_anomalies/len(labels)*100:.2f}%)")
    
    return model, scaler, labels, decision_scores


# ============================================================================
# UTILITY FUNCTIONS
# ============================================================================

def find_optimal_k_elbow_method(X, k_range=range(2, 11), random_state=42):
    """
    Find optimal number of clusters using Elbow Method (inertia/WSS).
    
    Args:
        X: Feature matrix
        k_range: Range of k values to test
        random_state: Random seed
    
    Returns:
        k_values, inertias, optimal_k (based on elbow)
    """
    from sklearn.cluster import KMeans
    
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    
    inertias = []
    k_values = list(k_range)
    
    for k in k_values:
        kmeans = KMeans(n_clusters=k, random_state=random_state, n_init=10)
        kmeans.fit(X_scaled)
        inertias.append(kmeans.inertia_)
    
    # Simple elbow detection (can be improved)
    # Find point with maximum curvature
    if len(inertias) > 2:
        # Calculate rate of change
        deltas = np.diff(inertias)
        deltas2 = np.diff(deltas)
        optimal_idx = np.argmax(np.abs(deltas2)) + 1
        optimal_k = k_values[optimal_idx]
    else:
        optimal_k = k_values[0]
    
    print(f"Optimal k (Elbow Method): {optimal_k}")
    
    return k_values, inertias, optimal_k


def find_optimal_k_silhouette_method(X, k_range=range(2, 11), random_state=42):
    """
    Find optimal number of clusters using Silhouette Method.
    
    Args:
        X: Feature matrix
        k_range: Range of k values to test
        random_state: Random seed
    
    Returns:
        k_values, silhouette_scores, optimal_k
    """
    from sklearn.cluster import KMeans
    
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    
    silhouette_scores = []
    k_values = list(k_range)
    
    for k in k_values:
        kmeans = KMeans(n_clusters=k, random_state=random_state, n_init=10)
        labels = kmeans.fit_predict(X_scaled)
        score = silhouette_score(X_scaled, labels)
        silhouette_scores.append(score)
    
    optimal_k = k_values[np.argmax(silhouette_scores)]
    
    print(f"Optimal k (Silhouette Method): {optimal_k} (score: {max(silhouette_scores):.4f})")
    
    return k_values, silhouette_scores, optimal_k


def evaluate_clustering(X, labels, true_labels=None):
    """
    Comprehensive evaluation of clustering results.
    
    Args:
        X: Feature matrix
        labels: Predicted cluster labels
        true_labels: True labels (if available, for external validation)
    
    Returns:
        Dictionary of evaluation metrics
    """
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    
    metrics = {}
    
    # Internal validation metrics
    metrics['silhouette_score'] = silhouette_score(X_scaled, labels)
    metrics['calinski_harabasz_score'] = calinski_harabasz_score(X_scaled, labels)
    metrics['davies_bouldin_score'] = davies_bouldin_score(X_scaled, labels)
    metrics['n_clusters'] = len(set(labels))
    
    # External validation metrics (if true labels available)
    if true_labels is not None:
        metrics['adjusted_rand_score'] = adjusted_rand_score(true_labels, labels)
        metrics['normalized_mutual_info_score'] = normalized_mutual_info_score(true_labels, labels)
    
    print("Clustering Evaluation Metrics:")
    for key, value in metrics.items():
        print(f"  {key}: {value:.4f}" if isinstance(value, float) else f"  {key}: {value}")
    
    return metrics


# ============================================================================
# EXAMPLE USAGE
# ============================================================================

if __name__ == "__main__":
    # Example: Create sample data
    from sklearn.datasets import make_blobs, make_moons, make_circles
    
    print("=" * 60)
    print("CLUSTERING EXAMPLES")
    print("=" * 60)
    
    # Generate sample clustering data
    X_blobs, y_blobs = make_blobs(
        n_samples=1000, n_features=2, centers=3,
        cluster_std=1.0, random_state=42
    )
    
    print("\n1. K-Means Clustering:")
    kmeans_model, kmeans_scaler, kmeans_labels, kmeans_centers = kmeans_example(
        X_blobs, n_clusters=3
    )
    
    print("\n2. Hierarchical Clustering:")
    hier_model, hier_scaler, hier_labels = hierarchical_clustering_example(
        X_blobs, n_clusters=3
    )
    
    print("\n3. DBSCAN:")
    dbscan_model, dbscan_scaler, dbscan_labels, dbscan_n_clusters = dbscan_example(
        X_blobs, eps=0.5, min_samples=5
    )
    
    print("\n4. Gaussian Mixture Model:")
    gmm_model, gmm_scaler, gmm_labels, gmm_probs = gmm_example(
        X_blobs, n_components=3
    )
    
    print("\n" + "=" * 60)
    print("DIMENSIONALITY REDUCTION EXAMPLES")
    print("=" * 60)
    
    # Generate high-dimensional data
    from sklearn.datasets import make_classification
    X_high_dim, _ = make_classification(
        n_samples=1000, n_features=20, n_informative=10,
        n_redundant=10, random_state=42
    )
    
    print("\n1. PCA:")
    pca_model, pca_scaler, pca_transformed, pca_variance = pca_example(
        X_high_dim, n_components=2
    )
    
    print("\n2. t-SNE:")
    tsne_model, tsne_scaler, tsne_transformed = tsne_example(
        X_high_dim, n_components=2, perplexity=30
    )
    
    print("\n3. ICA:")
    ica_model, ica_scaler, ica_transformed = ica_example(
        X_high_dim, n_components=2
    )
    
    print("\n" + "=" * 60)
    print("ANOMALY DETECTION EXAMPLES")
    print("=" * 60)
    
    # Generate data with outliers
    X_normal, _ = make_blobs(
        n_samples=900, n_features=2, centers=1,
        cluster_std=1.0, random_state=42
    )
    X_outliers = np.random.uniform(low=-10, high=10, size=(100, 2))
    X_with_outliers = np.vstack([X_normal, X_outliers])
    
    print("\n1. Isolation Forest:")
    iso_model, iso_scaler, iso_labels, iso_scores = isolation_forest_example(
        X_with_outliers, contamination=0.1
    )
    
    print("\n2. Local Outlier Factor:")
    lof_model, lof_scaler, lof_labels, lof_scores = lof_example(
        X_with_outliers, n_neighbors=20, contamination=0.1
    )
    
    print("\n" + "=" * 60)
    print("FINDING OPTIMAL NUMBER OF CLUSTERS")
    print("=" * 60)
    
    print("\n1. Elbow Method:")
    k_vals, inertias, optimal_k_elbow = find_optimal_k_elbow_method(X_blobs)
    
    print("\n2. Silhouette Method:")
    k_vals, sil_scores, optimal_k_sil = find_optimal_k_silhouette_method(X_blobs)
    
    print("\n3. Comprehensive Evaluation:")
    evaluate_clustering(X_blobs, kmeans_labels, true_labels=y_blobs)

