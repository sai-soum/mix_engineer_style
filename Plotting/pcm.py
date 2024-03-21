import os
import sys
import glob
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt 


from sklearn import preprocessing
from sklearn.manifold import TSNE
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
import plotly.express as px


def pca(X):
    pca = PCA(n_components=2)
    X_pca = pca.fit_transform(X)
    pca_labels = {str(i): f"PC {i+1} ({var:.1f}%)"
              for i, var in enumerate(pca.explained_variance_ratio_ * 100)}
    return X_pca, pca_labels

def tsne(X):
    tsne = TSNE(n_components=2, random_state=0)
    X_tsne = tsne.fit_transform(X)
    return X_tsne

def plot_pca_2D(components, label_df):
    fig = px.scatter(components, x=0, y=1, color=label_df)
    fig.write_image("images/pca.png")

def visualise_orig_dim(features, column_names, Y):
    fig = px.scatter_matrix(
        features,
        dimensions=column_names,
        color=Y,
    )
    fig.update_traces(diagonal_visible=False)
    fig.write_image("images/orig_dim.png")

def visualise_tsne(components, Y):
    fig = px.scatter(components, x=0, y=1, color=Y)
    fig.write_image("images/tsne.png")

def visualise_pc1_pc2(components, Y, pca_labels):

    
    fig = px.scatter_matrix(
    components,
    labels=pca_labels,
    dimensions=range(2),
    color=Y
)
    fig.update_traces(diagonal_visible=False)
    fig.write_image("images/pc1_pc2.png")

if __name__ == "__main__":

    #load the dataframes
    df = pd.read_csv('/homes/ssv02/mix_engineer_style/data/med_mix_features.csv')
    df = df.dropna()
    #labels
    Y = df['eng_name']
    song_name = df['songname']
    #features

     # Using drop() function to delete first n columns
    X = df[:].drop(['s_no','eng_name','songname'], axis = 1)
    columns = X.columns
    print(columns)
    X = StandardScaler().fit_transform(X) # normalizing the features
    X_df = pd.DataFrame(X, columns = columns) 
    X_PCA, pca_labels = pca(X_df)
    print(type(X_PCA))
    print(type(Y))
    print(Y[:10])
    print(X_PCA.shape)
    print(Y.shape)
    # if not os.path.exists('images'):
    #     os.makedirs('images')
    # plot_pca_2D(X_PCA, Y)
    # visualise_orig_dim(df, columns, Y)
    # visualise_pc1_pc2(X_PCA, Y, pca_labels)

    # #tsne
    # X_tsne = tsne(X_df)
    # visualise_tsne(X_tsne, Y)

    #consider_genre effect

    


