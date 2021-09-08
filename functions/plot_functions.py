#!/usr/bin/env python
# coding: utf-8

# In[4]:


# -*- coding: utf-8 -*-
"""
Created on Tue May  4 17:39:59 2021

Collection of custom evaluation functions for embedding

@author: marathomas
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from matplotlib.legend import Legend
import matplotlib
import seaborn as sns
import plotly.express as px
import plotly.graph_objects as go



def umap_2Dplot(x,y, scat_labels, mycolors, outname=None, showlegend=True):
    """
    Function that creates (and saves) 2D plot from an
    input dataset, color-colored by the provided labels.

    Parameters
    ----------
    x : 1D numpy array (numeric) or list
        x coordinates of datapoints
                
    y: 1D numpy array (numeric) or list 
       y coordinates of datapoints
    
    scat_labels: List-of-Strings
                 Datapoint labels
                     
    mycolors: String or List-of-Strings
              Seaborn color palette name (e.g. "Set2") or list of
              colors (Hex value strings) used for coloring datapoints
              (e.g. ["#FFEBCD","#0000FF",...])
                     
    outname: String
             Output filename at which plot will be saved
             No plot will be saved if outname is None
             (e.g. "my_folder/my_img.png")
             
    showlegend: Boolean
                Show legend if True, else don't
                
    Returns
    -------
    
    -
             
    """
    
    labeltypes = sorted(list(set(scat_labels)))
    pal = sns.color_palette(mycolors, n_colors=len(labeltypes))
    color_dict = dict(zip(labeltypes, pal))
    c = [color_dict[val] for val in scat_labels]
    
    fig = plt.figure(figsize=(6,6))
    
    plt.scatter(x, y, alpha=1,
                s=10, c=c)
    plt.xlabel('UMAP1')
    plt.ylabel('UMAP2');

    scatters = []
    for label in labeltypes:
        scatters.append(matplotlib.lines.Line2D([0],[0], linestyle="none", c=color_dict[label], marker = 'o'))
    
    if showlegend: plt.legend(scatters, labeltypes, numpoints = 1) 
    if outname: plt.savefig(outname, facecolor="white")



def umap_3Dplot(x,y,z,scat_labels, mycolors,outname=None, showlegend=True):
    """
    Function that creates (and saves) 3D plot from an
    input dataset, color-colored by the provided labels.

    Parameters
    ----------
    x : 1D numpy array (numeric) or list
        x coordinates of datapoints
                
    y: 1D numpy array (numeric) or list 
       y coordinates of datapoints

    z: 1D numpy array (numeric) or list 
       z coordinates of datapoints
    
    scat_labels: List-of-Strings
                 Datapoint labels
                     
    mycolors: String or List-of-Strings
              Seaborn color palette name (e.g. "Set2") or list of
              colors (Hex value strings) used for coloring datapoints
              (e.g. ["#FFEBCD","#0000FF",...])
                     
    outname: String
             Output filename at which plot will be saved
             No plot will be saved if outname is None
             (e.g. "my_folder/my_img.png")
             
    showlegend: Boolean
                Show legend if True, else don't
                
    Returns
    -------
    
    -
             
    """    
    labeltypes = sorted(list(set(scat_labels)))
    pal = sns.color_palette(mycolors, n_colors=len(labeltypes))
    color_dict = dict(zip(labeltypes, pal))
    c = [color_dict[val] for val in scat_labels]
    
    fig = plt.figure(figsize=(10,10))
    ax = fig.add_subplot(111, projection='3d')

    Axes3D.scatter(ax,
                    xs = x,
                    ys = y,
                    zs = z,
                    zdir='z',
                    s=20,
                    label = c,
                    c=c,
                    depthshade=False)

    ax.set_xlabel('UMAP1')
    ax.set_ylabel('UMAP2')
    ax.set_zlabel('UMAP3')
    
    ax.xaxis.pane.fill = False
    ax.yaxis.pane.fill = False
    ax.zaxis.pane.fill = False
    
    ax.xaxis.pane.set_edgecolor('w')
    ax.yaxis.pane.set_edgecolor('w')
    ax.zaxis.pane.set_edgecolor('w')



    if showlegend: 
        scatters = []
        for label in labeltypes:
            scatters.append(matplotlib.lines.Line2D([0],[0], linestyle="none", c=color_dict[label], marker = 'o'))
        
        ax.legend(scatters, labeltypes, numpoints = 1)
    
    if outname: plt.savefig(outname, facecolor="white")



def plotly_viz(x,y,z,scat_labels, mycolors):
    """
    Function that creates interactive 3D plot with plotly from
    an input dataset, color-colored by the provided labels.

    Parameters
    ----------
    x : 1D numpy array (numeric) or list
        x coordinates of datapoints
                
    y: 1D numpy array (numeric) or list 
       y coordinates of datapoints

    z: 1D numpy array (numeric) or list 
       z coordinates of datapoints
    
    scat_labels: List-of-Strings
                 Datapoint labels
                     
    mycolors: String or List-of-Strings
              Seaborn color palette name (e.g. "Set2") or list of
              colors (Hex value strings) used for coloring datapoints
              (e.g. ["#FFEBCD","#0000FF",...])
                
    Returns
    -------
    
    -
             
    """     
    labeltypes = sorted(list(set(scat_labels)))
    pal = sns.color_palette(mycolors, n_colors=len(labeltypes))
    color_dict = dict(zip(labeltypes, pal))
    c = [color_dict[val] for val in scat_labels]

    fig = go.Figure(data=[go.Scatter3d(x=x, y=y, z=z,
                                    mode='markers',
                                    hovertext = scat_labels,
                                    marker=dict(
                                        size=4,
                                        color=c,                # set color to an array/list of desired values
                                        opacity=0.8
                                        ))])

    fig.update_layout(scene = dict(
                      xaxis_title='UMAP1',
                      yaxis_title='UMAP2',
                      zaxis_title='UMAP3'),
                      width=700,
                      margin=dict(r=20, b=10, l=10, t=10))

    return fig


