#!/usr/bin/env python
# coding: utf-8

# In[4]:


# -*- coding: utf-8 -*-
"""
Created on Tue May  4 17:39:59 2021

Collection of custom evaluation functions for embedding

@author: marathomas
"""

import numpy as np
import pandas as pd
from sklearn.neighbors import NearestNeighbors
from sklearn.metrics import silhouette_samples, silhouette_score
import seaborn as sns
import matplotlib.pyplot as plt
import string
from scipy.spatial.distance import pdist, squareform
import sklearn
from sklearn.metrics.pairwise import euclidean_distances  


def make_nn_stats_dict(calltypes, labels, nb_indices):
    """
    Function that evaluates the labels of the k nearest neighbors of 
    all datapoints in a dataset.

    Parameters
    ----------
    calltypes : 1D numpy array (string) or list of strings
                set of class labels
    labels: 1D numpy array (string) or list of strings
            vector/list of class labels in dataset
    nb_indices: 2D numpy array (numeric integer)
                Array I(X,k) containing the indices of the k nearest
                nearest neighbors for each datapoint X of a
                dataset

    Returns
    -------
    nn_stats_dict : dictionary[<class label string>] = 2D numpy array (numeric)
                    dictionary that contains one array for each type of label.
                    Given a label L, nn_stats_dict[L] contains an array A(X,Y), 
                    where Y is the number of class labels in the dataset and each
                    row X represents a datapoint of label L in the dataset.
                    A[i,j] is the number of nearest neighbors of datapoint i that
                    are of label calltypes[j].
                    
    Example
    -------
    >>> 

    """
    nn_stats_dict = {}
    
    for calltype in calltypes:
        # which datapoints in the dataset are of this specific calltype?
        # -> get their indices
        call_indices = np.asarray(np.where(labels==calltype))[0]
        
        # initialize array that can save the class labels of the k nearest
        # neighbors of all these datapoints
        calltype_counts = np.zeros((call_indices.shape[0],len(calltypes)))
        
        # for each datapoint
        for i,ind in enumerate(call_indices):
            # what are the indices of its k nearest neighbors
            nearest_neighbors = nb_indices[ind]
            # for eacht of these neighbors
            for neighbor in nearest_neighbors:
                # what is their label
                neighbor_label = labels[neighbor]
                # put a +1 in the array
                calltype_counts[i,np.where(np.asarray(calltypes)==neighbor_label)[0][0]] += 1 
        
        # save the resulting array in dictionary 
        # (1 array per calltype)
        nn_stats_dict[calltype] = calltype_counts 
  
    return nn_stats_dict

def get_knn(k,embedding):
    """
    Function that finds k nearest neighbors (based on 
    euclidean distance) for each datapoint in a multidimensional 
    dataset 

    Parameters
    ----------
    k : integer
        number of nearest neighbors
    embedding: 2D numpy array (numeric)
               a dataset E(X,Y) with X datapoints and Y dimensions

    Returns
    -------
    indices: 2D numpy array (numeric)
             Array I(X,k) containing the indices of the k nearest
             nearest neighbors for each datapoint X of the input
             dataset
             
    distances: 2D numpy array (numeric)
               Array D(X,k) containing the euclidean distance to each
               of the k nearest neighbors for each datapoint X of the 
               input dataset. D[i,j] is the euclidean distance of datapoint
               embedding[i,:] to its jth neighbor.
                    
    Example
    -------
    >>> 

    """

    # Find k nearest neighbors
    nbrs = NearestNeighbors(metric='euclidean',n_neighbors=k+1, algorithm='brute').fit(embedding)
    distances, indices = nbrs.kneighbors(embedding)

    # need to remove the first neighbor, because that is the datapoint itself
    indices = indices[:,1:]  
    distances = distances[:,1:]
    
    return indices, distances


def make_statstabs(nn_stats_dict, calltypes, labels,k):
    """
    Function that generates two summary tables containing
    the frequency of different class labels among the k nearest 
    neighbors of datapoints belonging to a class.

    Parameters
    ----------
    nn_stats_dict : dictionary[<class label string>] = 2D numpy array (numeric)
                    dictionary that contains one array for each type of label.
                    Given a label L, nn_stats_dict[L] contains an array A(X,Y), 
                    where Y is the number of class labels in the dataset and each
                    row X represents a datapoint of label L in the dataset.
                    A[i,j] is the number of nearest neighbors of datapoint i that
                    are of label calltypes[j].
                    (is returned from evaulation_functions.make_nn_statsdict)
    calltypes : 1D numpy array (string) or list of strings
                set of class labels
    labels: 1D numpy array (string) or list of strings
            vector/list of class labels in dataset
    k: Integer
       number of nearest neighbors

    Returns
    -------
    stats_tab: 2D pandas dataframe (numeric)
               Summary table T(X,Y) with X,Y = number of classes.
               T[i,j] is the average percentage of datapoints with class label j
               in the neighborhood of datapoints with class label i
             
    stats_tab_norm: 2D pandas dataframe (numeric)
                   Summary table N(X,Y) with X,Y = number of classes.
                   N[i,j] is the log2-transformed ratio of the percentage of datapoints 
                   with class label j in the neighborhood of datapoints with class label i
                   to the percentage that would be expected by random chance and random
                   distribution. (N[i,j] = log2(T[i,j]/random_expect))
              
    Example
    -------
    >>> 

    """
    
    # Get the class frequencies in the dataset
    overall = np.zeros((len(calltypes)))  
    for i,calltype in enumerate(calltypes):
        overall[i] = sum(labels==calltype) 
    overall = (overall/np.sum(overall))*100
    
    # Initialize empty array for stats_tab and stats_tab_norm
    stats_tab = np.zeros((len(calltypes),len(calltypes)))
    stats_tab_norm = np.zeros((len(calltypes),len(calltypes)))

    # For each calltype
    for i, calltype in enumerate(calltypes):
        # Get the table with all neighbor label counts per datapoint
        stats = nn_stats_dict[calltype]
        # Average across all datapoints and transform to percentage
        stats_tab[i,:] = (np.mean(stats,axis=0)/k)*100
        # Divide by overall percentage of this class in dataset 
        # for the normalized statstab version
        stats_tab_norm[i,:] = ((np.mean(stats,axis=0)/k)*100)/overall
    
    # Turn into dataframe
    stats_tab = pd.DataFrame(stats_tab)
    stats_tab_norm = pd.DataFrame(stats_tab_norm)
    
    # Add row with overall frequencies to statstab
    stats_tab.loc[len(stats_tab)] = overall
    
    # Name columns and rows
    stats_tab.columns = calltypes
    stats_tab.index = calltypes+['overall']

    stats_tab_norm.columns = calltypes
    stats_tab_norm.index = calltypes
    
    # Replace zeros with small value as otherwise log2 transform cannot be applied
    x=stats_tab_norm.replace(0, 0.0001)
    
    # log2-tranform the ratios that are currently in statstabnorm
    stats_tab_norm = np.log2(x)

    return stats_tab, stats_tab_norm


class nn:
    """
    A class to represent nearest neighbor statistics for a
    given latent space representation of a labelled dataset

    Attributes
    ----------
    embedding : 2D numpy array (numeric)
                a dataset E(X,Y) with X datapoints and Y dimensions
                
    labels: 1D numpy array (string) or list of strings
            vector/list of class labels in dataset
    k : integer
        number of nearest neighbors to consider
    
    statstab: 2D pandas dataframe (numeric)
               Summary table T(X,Y) with X,Y = number of classes.
               T[i,j] is the average percentage of datapoints with class label j
               in the neighborhood of datapoints with class label i
             
    statstabnorm: 2D pandas dataframe (numeric)
                   Summary table N(X,Y) with X,Y = number of classes.
                   N[i,j] is the log2-transformed ratio of the percentage of datapoints 
                   with class label j in the neighborhood of datapoints with class label i
                   to the percentage that would be expected by random chance and random
                   distribution. (N[i,j] = log2(T[i,j]/random_expect))      

    Methods
    -------
    
    def knn_cc():
        returns k nearest neighbor fractional consistency for each class
        (1D numpy array). What percentage of datapoints (of this class)
        have fully consistent k neighbors (all k are also of the same class)
    
    def knn_accuracy(self):
        returns k nearest neighbor classifier accuracy for each class
        (1D numpy array). What percentage of datapoints (of this class)
        have a majority of same-class neighbors among k nearest neighbors
    
    get_statstab():
        returns statstab
    
    get_statstabnorm():
        returns statstabnorm
    
    get_S():    
        returns S score of embedding
        S(class X) is the average percentage of same-class neighbors
        among the k nearest neighbors of all datapoints of
        class X. S of an embedding is the average of S(class X) over all
        classes X (unweighted, e.g. does not consider class frequencies).
    
    get_Snorm():
        returns Snorm score of embedding
        Snorm(class X) is the log2 transformed, normalized percentage of 
        same-class neighbors among the k nearest neighbors of all datapoints of
        class X. Snorm of an embedding is the average of Snorm(class X) over all
        classes X.
    
    get_ownclass_S():
        returns array of S(class X) score for each class X in the dataset
        (alphanumerically sorted by class name)
        S(class X) is the average percentage of same-class neighbors
        among the k nearest neighbors of all datapoints of
        class X.
    
    get_ownclass_Snorm():
        returns array of Snorm(class X) score for each class X in the dataset
        (alphanumerically sorted by class name)
        Snorm(class X) is the log2 transformed, normalized percentage of 
        same-class neighbors among the k nearest neighbors of all datapoints of
        class X. 
    
    plot_heat_S(vmin, vmax, center, cmap, cbar, outname)
        plots heatmap of S scores
    
    plot_heat_S(vmin, vmax, center, cmap, cbar, outname)
        plots heatmap of Snorm scores
        
    plot_heat_S(center, cmap, cbar, outname)
        plots heatmap of fold likelihood (statstabnorm scores to the power of 2)
        
    draw_simgraph(outname)
        draws similarity graph based on statstabnorm scores
    
    """
    def __init__(self, embedding, labels, k):
        
        self.embedding = embedding
        self.labels = labels
        self.k = k
        
        label_types = sorted(list(set(labels)))        
        
        indices, distances = get_knn(k,embedding)
        nn_stats_dict = make_nn_stats_dict(label_types, labels, indices)
        stats_tab, stats_tab_norm = make_statstabs(nn_stats_dict, label_types, labels, k)
        
        self.nn_stats_dict = nn_stats_dict
        self.statstab = stats_tab
        self.statstabnorm = stats_tab_norm
    
    def knn_cc(self):
        label_types = sorted(list(set(self.labels)))        
        consistent = []
        for i,labeltype in enumerate(label_types):
            statsd = self.nn_stats_dict[labeltype] 
            x = statsd[:,i]
            cc = (np.sum(x == self.k) / statsd.shape[0])*100
            consistent.append(cc)
        return np.asarray(consistent)
    
    def knn_accuracy(self):
        label_types = sorted(list(set(self.labels)))        
        has_majority = []
        if (self.k % 2) == 0:
            n_majority = (self.k/2)+ 1
        else:
            n_majority = (self.k/2)+ 0.5
        for i,labeltype in enumerate(label_types):
            statsd = self.nn_stats_dict[labeltype] 
            x = statsd[:,i]
            cc = (np.sum(x >= n_majority) / statsd.shape[0])*100  
            has_majority.append(cc)
        return np.asarray(has_majority)  
          
    def get_statstab(self):
        return self.statstab
    
    def get_statstabnorm(self):
        return self.statstabnorm
    
    def get_S(self):    
        return np.mean(np.diagonal(self.statstab))
    
    def get_Snorm(self):
        return np.mean(np.diagonal(self.statstabnorm))
    
    def get_ownclass_S(self):
        return np.diagonal(self.statstab)
    
    def get_ownclass_Snorm(self):
        return np.diagonal(self.statstabnorm)
    
    def plot_heat_S(self,vmin=0, vmax=100, center=50, cmap=sns.color_palette("Greens", as_cmap=True), cbar=None, outname=None):
        plt.figure(figsize=(6,6))
        ax=sns.heatmap(self.statstab, annot=True, vmin=vmin, vmax=vmax, center=center, cmap=cmap, cbar=cbar)
        plt.xlabel("neighbor label")
        plt.ylabel("datapoint label")
        plt.title("Nearest Neighbor Frequency P")
        if outname:
            plt.savefig(outname, facecolor="white")

    def plot_heat_Snorm(self,vmin=-13, vmax=13, center=0, cmap=sns.diverging_palette(20, 145, as_cmap=True), cbar=None, outname=None):
        plt.figure(figsize=(6,6))
        ax=sns.heatmap(self.statstabnorm, annot=True, vmin=vmin, vmax=vmax, center=center, cmap=cmap, cbar=cbar)
        plt.xlabel("neighbor label")
        plt.ylabel("datapoint label")
        plt.title("Normalized Nearest Neighbor Frequency Pnorm")
        if outname:
            plt.savefig(outname, facecolor="white")
    
    def plot_heat_fold(self, center=1, cmap=sns.diverging_palette(20, 145, as_cmap=True), cbar=None, outname=None):
        plt.figure(figsize=(6,6))
        ax=sns.heatmap(np.power(2,self.statstabnorm), annot=True, center=center, cmap=cmap, cbar=cbar)
        plt.xlabel("neighbor label")
        plt.ylabel("datapoint label")
        plt.title("Nearest Neighbor fold likelihood")
        if outname:
            plt.savefig(outname, facecolor="white")
            
    def draw_simgraph(self, outname="simgraph.png"):
        
        # Imports here because specific to this method and
        # sometimes problematic to install (dependencies)

        import networkx as nx
        import pygraphviz
        
        calltypes = sorted(list(set(self.labels)))
        sim_mat = np.asarray(self.statstabnorm).copy()
        for i in range(sim_mat.shape[0]):
            for j in range(i,sim_mat.shape[0]):
                if i!=j:
                    sim_mat[i,j] = np.mean((sim_mat[i,j], sim_mat[j,i]))
                    sim_mat[j,i] = sim_mat[i,j]
                else:
                    sim_mat[i,j] = 0
                    
        dist_mat = sim_mat*(-1)
        dist_mat = np.interp(dist_mat, (dist_mat.min(), dist_mat.max()), (1, 10))
        
        for i in range(dist_mat.shape[0]):
            dist_mat[i,i] = 0
            
        dt = [('len', float)]
        
        A = dist_mat
        A = A.view(dt)

        G = nx.from_numpy_matrix(A)
        G = nx.relabel_nodes(G, dict(zip(range(len(G.nodes())),calltypes))) 

        G = nx.drawing.nx_agraph.to_agraph(G)

        G.node_attr.update(color="#bec1d4", style="filled", shape='circle', fontsize='20')
        G.edge_attr.update(color="blue", width="2.0")
        print("Graph saved at ", outname)
        G.draw(outname, format='png', prog='neato')
        return G

    
    
class sil:
    """
    A class to represent Silhouette score statistics for a
    given latent space representation of a labelled dataset

    Attributes
    ----------
    embedding : 2D numpy array (numeric)
                a dataset E(X,Y) with X datapoints and Y dimensions
                
    labels: 1D numpy array (string) or list of strings
            vector/list of class labels in dataset

    
    labeltypes: list of strings
                 alphanumerically sorted set of class labels
                 
    avrg_SIL: Numeric (float)
              The average Silhouette score of the dataset
             
    sample_SIL: 1D numpy array (numeric)
                The Silhouette scores for each datapoint in the dataset
    
    Methods
    -------
    
    get_avrg_score():
        returns the average Silhouette score of the dataset
    
    get_score_per_class():
        returns the average Silhouette score per class for each
        class in the dataset as 1D numpy array
        (alphanumerically sorted classes)
        
    get_sample_scores():
        returns the Silhouette scores for each datapoint in the dataset
        (1D numpy array, numeric)
             
    
    """
    def __init__(self, embedding, labels):
        
        self.embedding = embedding
        self.labels = labels
        self.labeltypes = sorted(list(set(labels)))
        
        self.avrg_SIL = silhouette_score(embedding, labels)
        self.sample_SIL = silhouette_samples(embedding, labels)
    
    def get_avrg_score(self):
        return self.avrg_SIL
    
    def get_score_per_class(self):
        scores = np.zeros((len(self.labeltypes),))
        for i, label in enumerate(self.labeltypes):
            ith_cluster_silhouette_values = self.sample_SIL[self.labels == label]
            scores[i] = np.mean(ith_cluster_silhouette_values)
            #scores_tab = pd.DataFrame([scores],columns=self.labeltypes)
        return scores
    
    def get_sample_scores(self):
        return self.sample_SIL
    
    def plot_sil(self, mypalette="Set2", embedding_type=None, outname=None):
        labeltypes = sorted(list(set(self.labels)))
        n_clusters = len(labeltypes)

        # Create a subplot with 1 row and 2 columns
        fig, ax1 = plt.subplots(1, 1)
        fig.set_size_inches(9, 7)
        ax1.set_xlim([-1, 1])
        ax1.set_ylim([0, self.embedding.shape[0] + (n_clusters + 1) * 10])
        y_lower = 10
        
        pal = sns.color_palette(mypalette, n_colors=len(labeltypes))
        color_dict = dict(zip(labeltypes, pal))
        
        labeltypes = sorted(labeltypes, reverse=True)


        for i, cluster_label in enumerate(labeltypes):
            ith_cluster_silhouette_values = self.sample_SIL[self.labels == cluster_label]
            ith_cluster_silhouette_values.sort()

            size_cluster_i = ith_cluster_silhouette_values.shape[0]
            y_upper = y_lower + size_cluster_i

            ax1.fill_betweenx(np.arange(y_lower, y_upper),
                            0, ith_cluster_silhouette_values,
                            facecolor=color_dict[cluster_label], edgecolor=color_dict[cluster_label], alpha=0.7)

            ax1.text(-0.05, y_lower + 0.5 * size_cluster_i, cluster_label)

            # Compute the new y_lower for next plot
            y_lower = y_upper + 10  # 10 for the 0 samples
        
        if embedding_type:
            mytitle = "Silhouette plot for "+embedding_type+" labels"
        else:
            mytitle = "Silhouette plot"

        ax1.set_title(mytitle)
        ax1.set_xlabel("Silhouette value")
        ax1.set_ylabel("Cluster label")

        # The vertical line for average silhouette score of all the values
        ax1.axvline(x=self.avrg_SIL, color="red", linestyle="--")
        
        if outname:
            plt.savefig(outname, facecolor="white")


            
            
def plot_within_without(embedding,labels, distance_metric = "euclidean", outname=None,xmin=0, xmax=12, ymax=0.5, nbins=50,nrows=4, ncols=2, density=True):
    """
    Function that plots distribution of pairwise distances within a class
    vs. towards other classes ("between"), for each class in a dataset

    Parameters
    ----------
    embedding : 2D numpy array (numeric)
                a dataset E(X,Y) with X datapoints and Y dimensions
                
    labels: 1D numpy array (string) or list of strings
            vector/list of class labels in dataset

    
    distance_metric: String
                     Type of distance metric, e.g. "euclidean", "manhattan"...
                     all scipy.spatial.distance metrics are allowed
                     
    outname: String
             Output filename at which plot will be saved
             No plot will be saved if outname is None
             (e.g. "my_folder/my_img.png")
             
    xmin, xmax: Numeric
                Min and max of x-axis
    
    ymax: Numeric
          Max of yaxis
    
    nbins: Integer
           Number of bins in histograms
    
    nrows: Integer
           Number of rows of subplots
    
    ncols: Integer
           Number of columns of subplots
    
    density: Boolean
             Plot density histogram if density=True
             else plot frequency histogram
                
    Returns
    -------
    
    -
             
    """
    
    distmat_embedded = squareform(pdist(embedding, metric=distance_metric))
    labels = np.asarray(labels)
    calltypes = sorted(list(set(labels)))

    self_dists={}
    other_dists={}

    for calltype in calltypes:
        x=distmat_embedded[np.where(labels==calltype)]
        x = np.transpose(x)  
        y = x[np.where(labels==calltype)]

        self_dists[calltype] = y[np.triu_indices(n=y.shape[0], m=y.shape[1],k = 1)]
        y = x[np.where(labels!=calltype)]
        other_dists[calltype] = y[np.triu_indices(n=y.shape[0], m=y.shape[1], k = 1)]
    
    plt.figure(figsize=(8, 8))
    i=1

    for calltype in calltypes:

        plt.subplot(nrows, ncols, i)
        n, bins, patches = plt.hist(x=self_dists[calltype], label="within", density=density,
                                  bins=np.linspace(xmin, xmax, nbins), color='green',
                                  alpha=0.5, rwidth=0.85)

        plt.vlines(x=np.mean(self_dists[calltype]),ymin=0,ymax=ymax,color='green', linestyles='dotted')

        n, bins, patches = plt.hist(x=other_dists[calltype], label="between", density=density,
                                  bins=np.linspace(xmin, xmax, nbins), color='red',
                                  alpha=0.5, rwidth=0.85)

        plt.vlines(x=np.mean(other_dists[calltype]),ymin=0,ymax=ymax,color='red', linestyles='dotted')
        plt.legend()
        plt.grid(axis='y', alpha=0.75)
        plt.title(calltype)
        plt.xlim(xmin,xmax)
        plt.ylim(0, ymax)
        
        if (i%ncols)==1:
            ylabtitle = 'Density' if density else 'Frequency'
            plt.ylabel(ylabtitle)
        if i>=((nrows*ncols)-ncols):
            plt.xlabel(distance_metric+' distance')

        i=i+1

    plt.tight_layout()
    if outname:
        plt.savefig(outname, facecolor="white")
    

def next_sameclass_nb(embedding, labels):
    """
    Function that calculates the neighborhood degree of the closest 
    same-class neighbor for a given labelled dataset. Calculation is
    based on euclidean distance and done for each datapoint. E.g. 6 
    means that the 6th nearest neighbor of this datapoint is the first
    to be of the same-class (the first 5 nearest neighbors are of
    different class)

    Parameters:
    ----------
    embedding : 2D numpy array (numeric)
                a dataset E(X,Y) with X datapoints and Y dimensions
                
    labels: 1D numpy array (string) or list of strings
            vector/list of class labels in dataset
    
    Returns:
    -------
    
    nbs_to_sameclass: 1D numpy array
                      nearest same-class neighborhood degree for
                      each datapoint of the input dataset
    
    """
    indices = []
    distmat = euclidean_distances(embedding, embedding)
    k = embedding.shape[0]-1
    
    nbs_to_sameclass = []

    for i in range(distmat.shape[0]):
        neighbors = []
        distances = distmat[i,:]
        ranks = np.array(distances).argsort().argsort()
        for j in range(1,embedding.shape[0]):
            ind = np.where(ranks==j)[0]
            nb_label = labels[ind[0]]
            neighbors.append(nb_label)
        
        neighbors = np.asarray(neighbors)
            
        # How many neighbors until I encounter a same-class neighbor?
        own_type = labels[i]
        distances = distmat[i,:]
        ranks = np.array(distances).argsort().argsort()
        neighbors = []
        for j in range(1,embedding.shape[0]):
            ind = np.where(ranks==j)[0]
            nb_label = labels[ind[0]]
            neighbors.append(nb_label)
        
        neighbors = np.asarray(neighbors)
            
        # How long to same-class label?
        own_type = labels[i]
        first_occurrence = np.where(neighbors==labels[i])[0][0]
    
        nbs_to_sameclass.append(first_occurrence)
    
    nbs_to_sameclass = np.asarray(nbs_to_sameclass)
    return(nbs_to_sameclass)



