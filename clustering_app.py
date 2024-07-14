import streamlit as st
import numpy as np
import pandas as pd
from sklearn.datasets import load_iris
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans, AgglomerativeClustering, DBSCAN
from sklearn.metrics import silhouette_score, davies_bouldin_score
import seaborn as sns
import matplotlib.pyplot as plt

# Function to load and preprocess dataset
def load_data(file):
    if file is None:
        return None
    df = pd.read_csv(file)
    return df

# Function for K-means clustering
def kmeans_clustering(data, n_clusters):
    scaler = StandardScaler()
    data_scaled = scaler.fit_transform(data)
    kmeans = KMeans(n_clusters=n_clusters, random_state=42)
    labels = kmeans.fit_predict(data_scaled)
    return labels, kmeans.cluster_centers_

# Function for hierarchical clustering
def hierarchical_clustering(data, n_clusters):
    scaler = StandardScaler()
    data_scaled = scaler.fit_transform(data)
    hierarchical = AgglomerativeClustering(n_clusters=n_clusters)
    labels = hierarchical.fit_predict(data_scaled)
    return labels

# Function for DBSCAN clustering
def dbscan_clustering(data, eps, min_samples):
    scaler = StandardScaler()
    data_scaled = scaler.fit_transform(data)
    dbscan = DBSCAN(eps=eps, min_samples=min_samples)
    labels = dbscan.fit_predict(data_scaled)
    return labels

# Main function for Streamlit app
st.title('LIBRARY MANAGEMENT SYSTEM')
def main():
    st.title('Clustering Analysis')

    # File uploader for dataset
    file = st.file_uploader('Upload dataset (CSV)', type=['csv'])
    if file is not None:
        df = load_data(file)
        st.write('Preview of dataset:')
        st.write(df.head())

        # Select clustering algorithm
        algorithm = st.selectbox('Select Clustering Algorithm', ['K-means', 'Hierarchical', 'DBSCAN'])

        if algorithm == 'K-means':
            n_clusters = st.number_input('Number of clusters', min_value=2, max_value=10, value=3)
            if st.button('Run K-means Clustering'):
                labels, centers = kmeans_clustering(df, n_clusters)
                st.write(f'Cluster centers:\n{centers}')

        elif algorithm == 'Hierarchical':
            n_clusters = st.number_input('Number of clusters', min_value=2, max_value=10, value=3)
            if st.button('Run Hierarchical Clustering'):
                labels = hierarchical_clustering(df, n_clusters)

        elif algorithm == 'DBSCAN':
            eps = st.slider('EPS', min_value=0.1, max_value=1.0, value=0.5, step=0.1)
            min_samples = st.slider('Min Samples', min_value=1, max_value=10, value=5)
            if st.button('Run DBSCAN Clustering'):
                labels = dbscan_clustering(df, eps, min_samples)

        # Show clustering results
        if 'labels' in locals():
            st.write('Cluster Labels:')
            st.write(labels)

            # Evaluate clustering
            silhouette = silhouette_score(df, labels)
            davies_bouldin = davies_bouldin_score(df, labels)
            st.write(f'Silhouette Score: {silhouette}')
            st.write(f'Davies-Bouldin Index: {davies_bouldin}')

            # Visualize clustering results
            st.write('Clustering Visualization:')
            plt.figure(figsize=(10, 7))
            sns.scatterplot(data=df, x='feature1', y='feature2', hue=labels, palette='viridis')
            st.pyplot()

if __name__ == '__main__':
    main()
