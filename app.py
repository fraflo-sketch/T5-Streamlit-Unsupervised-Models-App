import streamlit as st
import pandas as pd
from models import (
    preprocess_data,
    apply_feature_selection,
    run_clustering,
    evaluate_clustering,
    plot_clusters
)

st.set_page_config(page_title="Clustering App", layout="centered")

st.title("Clustering App")

uploaded_file = st.file_uploader("Upload your dataset", type=["csv"])

if uploaded_file is not None:
    df = pd.read_csv(uploaded_file)
    
    st.subheader("Dataset Preview")
    st.dataframe(df.head())
    
    
    df_processed = preprocess_data(df)
    
    st.write("")
    feature_mode = st.radio("Select feature mode:", ["Manual Selection", "PCA Reduction"], horizontal=True)
    
    if feature_mode == "Manual Selection":
        selected_features = st.multiselect(
            "Select the features for clustering",
            options=df_processed.columns.tolist(),
            default=df_processed.columns.tolist()[:2] if len(df_processed.columns) >= 2 else df_processed.columns.tolist()
        )
        if len(selected_features) < 2:
            st.warning("Selecciona al menos 2 características para continuar.")
            st.stop()
        X_input, feat_names = apply_feature_selection(df_processed, feature_mode, selected_features=selected_features)
    else:
        max_comps = min(df_processed.shape[1], df_processed.shape[0])
        n_components = st.number_input(
            "Number of PCA components (min 2):",
            min_value=2,
            max_value=max_comps if max_comps >= 2 else 2,
            value=min(2, max_comps)
        )
        selected_features = []
        X_input, feat_names = apply_feature_selection(df_processed, feature_mode, n_components=n_components)

    st.header("Model Selection")
    algorithm = st.selectbox(
        "Select a clustering algorithm",
        ["K-Means", "K-Medoids", "DBSCAN", "Hierarchical Clustering"]
    )
    
    params = {}
    if algorithm == "K-Means":
        params['n_clusters'] = st.slider("n_clusters", min_value=2, max_value=10, value=3)
    elif algorithm == "K-Medoids":
        params['n_clusters'] = st.slider("n_clusters", min_value=2, max_value=10, value=3)
        params['metric'] = st.selectbox("metric", ["euclidean", "cityblock", "cosine"])
    elif algorithm == "DBSCAN":
        col_ep, col_min = st.columns(2)
        with col_ep:
            params['eps'] = st.number_input("eps", min_value=0.01, max_value=10.0, value=0.5, step=0.1)
        with col_min:
            params['min_samples'] = st.number_input("min_samples", min_value=1, max_value=50, value=5)
    elif algorithm == "Hierarchical Clustering":
        params['n_clusters'] = st.slider("n_clusters", min_value=2, max_value=10, value=3)
        params['linkage'] = st.selectbox("linkage", ["ward", "complete", "average", "single"])

    labels, model_obj = run_clustering(algorithm, X_input, **params)

    st.header("Clustered Dataset")
    df_output = df.copy()
    df_output['Cluster'] = labels
    st.dataframe(df_output)

    sil, db = evaluate_clustering(X_input, labels)
    if sil is not None and db is not None:
        m1, m2 = st.columns(2)
        m1.metric("Silhouette Score", f"{sil:.4f}")
        m2.metric("Davies-Bouldin Index", f"{db:.4f}")

    
    fig = plot_clusters(df, X_input, labels, algorithm, feature_mode, selected_features)
    st.pyplot(fig)