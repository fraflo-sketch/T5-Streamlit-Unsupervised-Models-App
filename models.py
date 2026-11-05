import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.spatial.distance import cdist
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.decomposition import PCA
from sklearn.cluster import KMeans, DBSCAN, AgglomerativeClustering
from sklearn.metrics import silhouette_score, davies_bouldin_score

class NativeKMedoids:
    def __init__(self, n_clusters=3, metric='euclidean', max_iter=100, random_state=42):
        self.n_clusters = n_clusters
        self.metric = metric
        self.max_iter = max_iter
        self.random_state = random_state

    def fit_predict(self, X):
        np.random.seed(self.random_state)
        n_samples = X.shape[0]
        dist_matrix = cdist(X, X, metric=self.metric)
        medoid_indices = np.random.choice(n_samples, self.n_clusters, replace=False)
        
        for _ in range(self.max_iter):
            labels = np.argmin(dist_matrix[:, medoid_indices], axis=1)
            new_medoids = np.copy(medoid_indices)
            for k in range(self.n_clusters):
                members = np.where(labels == k)[0]
                if len(members) > 0:
                    sub_dist = dist_matrix[np.ix_(members, members)]
                    new_medoids[k] = members[np.argmin(sub_dist.sum(axis=1))]
            if np.array_equal(medoid_indices, new_medoids):
                break
            medoid_indices = new_medoids
            
        return np.argmin(dist_matrix[:, medoid_indices], axis=1)

def preprocess_data(df: pd.DataFrame) -> pd.DataFrame:
    df_processed = df.copy()
    
    categorical_cols = df_processed.select_dtypes(include=['object', 'category']).columns
    for col in categorical_cols:
        le = LabelEncoder()
        df_processed[col] = le.fit_transform(df_processed[col].astype(str))
        
    numerical_cols = df_processed.select_dtypes(include=['int64', 'float64', 'int32', 'float32']).columns
    if len(numerical_cols) > 0:
        scaler = StandardScaler()
        df_processed[numerical_cols] = scaler.fit_transform(df_processed[numerical_cols])
        
    return df_processed

def apply_feature_selection(df_processed: pd.DataFrame, mode: str, selected_features=None, n_components=2):
    if mode == "Manual Selection":
        X_input = df_processed[selected_features].values
        names = selected_features
    else:
        pca = PCA(n_components=n_components, random_state=42)
        X_input = pca.fit_transform(df_processed)
        names = [f"PCA_{i+1}" for i in range(n_components)]
    return X_input, names

def run_clustering(algorithm: str, X_input: np.ndarray, **kwargs):
    if algorithm == "K-Means":
        model = KMeans(n_clusters=kwargs.get('n_clusters', 3), random_state=42, n_init=10)
    elif algorithm == "K-Medoids":
        model = NativeKMedoids(
            n_clusters=kwargs.get('n_clusters', 3),
            metric=kwargs.get('metric', 'euclidean'),
            random_state=42
        )
    elif algorithm == "DBSCAN":
        model = DBSCAN(
            eps=kwargs.get('eps', 0.5),
            min_samples=kwargs.get('min_samples', 5)
        )
    elif algorithm == "Hierarchical Clustering":
        model = AgglomerativeClustering(
            n_clusters=kwargs.get('n_clusters', 3),
            linkage=kwargs.get('linkage', 'ward')
        )
    else:
        raise ValueError(f"Algoritmo desconocido: {algorithm}")
        
    labels = model.fit_predict(X_input)
    return labels, model

def evaluate_clustering(X_input: np.ndarray, labels: np.ndarray):
    unique_labels = set(labels)
    non_noise = [l for l in unique_labels if l != -1]
    
    if len(unique_labels) > 1 and len(non_noise) > 0:
        try:
            sil = silhouette_score(X_input, labels)
            db = davies_bouldin_score(X_input, labels)
            return sil, db
        except Exception:
            return None, None
    return None, None

def plot_clusters(df_original: pd.DataFrame, X_input: np.ndarray, labels: np.ndarray, algorithm: str, mode: str, selected_features=None):
    fig, ax = plt.subplots(figsize=(6, 5))
    
    if mode == "Manual Selection" and selected_features and len(selected_features) == 2:
        x_data = df_original[selected_features[0]]
        y_data = df_original[selected_features[1]]
        x_label, y_label = selected_features[0], selected_features[1]
    else:
        if X_input.shape[1] >= 2:
            pca_vis = PCA(n_components=2, random_state=42)
            X_vis = pca_vis.fit_transform(X_input)
            x_data = X_vis[:, 0]
            y_data = X_vis[:, 1]
        else:
            x_data = X_input[:, 0]
            y_data = np.zeros_like(x_data)
        x_label, y_label = "Componente Principal 1", "Componente Principal 2"
        
    ax.scatter(x_data, y_data, c=labels, cmap='viridis', alpha=0.85, edgecolors='none')
    ax.set_xlabel(x_label)
    ax.set_ylabel(y_label)
    ax.set_title(algorithm, fontsize=14)
    ax.grid(True, linestyle='--', alpha=0.3)
    
    return fig