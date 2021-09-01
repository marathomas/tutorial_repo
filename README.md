# tutorial_repo
Tutorial for generating and evaluating latent-space representations of vocalizations using UMAP

## About

This tutorial contains a sequence of jupyter notebook files that help you generate latent space representations from input audio files.

## Structure

    ├── notebooks     
    │   ├── 01_generate_spectrograms.ipynb      
    │   ├── 02a_generate_UMAP_basic.ipynb       
    │   ├── 02b_generate_UMAP_timeshift.ipynb  
    │   ├── 03_UMAP_clustering.ipynb            
    │   ├── 03_UMAP_eval.ipynb                  
    │   ├── 03_UMAP_viz_part_1_prep.ipynb       
    │   └── 03_UMAP_viz_part_2_tool.ipynb        
    ├── audio                                  <- input soundfiles
    │   ├── call_1.wav     
    │   ├── call_2.wav     
    │   ├── call_3.wav     
    │   └── ...            
    ├── functions                         
    │   ├── audio_functions.py            
    │   ├── custom_dist_functions.py      
    │   ├── evaluation_functions.py       
    │   ├── plot_functions.py             
    │   └── preprocessing_functions.py    
    ├── data 
    │   └── info_file.csv                      <- input metadata file
    ├── environments 
    │   └── umap_tut_env.yaml
    ├── ... 
    
    
## Requirements

### Packages, installations etc.

A conda environment file is included in /environments. For manual install, these are the core packages:

>umap-learn
>librosa
>ipywidgets
>pandas=1.2.4
>seaborn
>pysoundfile=0.10.3
>voila
>graphviz
>hdbscan
>plotly
>networkx
>pygraphviz

Make sure to enable jupyter widgets with:
>jupyter nbextension enable --py widgetsnbextension


### Input

#### Audio files

You need a dataset of sound files, each containing a single vocalization or syllable as input (in /audio). 
(You may have to detect and extract such vocal elements first, if working with acoustic recordings.)

Ideally, start and end of the sound file correspond exactly to start and end of the vocalization. 
If there are delays in the onset of the vocalizations, these should be the same for all sound files. 
Otherwise, vocalizations may appear dissimilar or distant in latent space simply because their onset times are different. 
If it is not possible to mark the start times correctly, use the timeshift option to generate UMAP embeddings,
but note that it comes at the cost of increased computation time.

#### Info file

You need a ";"-separated .csv file with headers containing the filenames of the input audio, some labels and any other additional metadata (if avilable). 
If some or all labels are unknown, there should still be a label column and unkown labels should be marked with "unknown".

Structure of info_file.csv:

    | filename   | label   | ...    |  .... 
    -----------------------------------------
    | call_1.wav | alarm   |  ...   |  ....   
    | call_2.wav | contact |  ...   |  ....  
    | ...        |  ...    |  ...   |  ....   


## Structure

### 1. Start with 01_generate_spectrograms.ipynb to generate spectrograms from input audio files.
### 2. Generate latent space representations with 02a_generate_UMAP_basic.ipynb OR 02b_generate_UMAP_timeshift.ipynb 

### 3. You can now 
- evaluate the latent space representation with 03_UMAP_eval.ipynb,
 
- visualize it by running 03_UMAP_viz_part_1_prep.ipynb and 03_UMAP_viz_part_2_tool.ipynb or

- apply clustering on the latent space representation with 03_UMAP_clustering.ipynb 


## Data accessibility

This data is part of an ongoing study and is protected by copyright law, meaning that no one may reproduce, distribute, or create derivative works from it. If you want to access data, please get in touch.
