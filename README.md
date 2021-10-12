# Tutorial for generating and evaluating latent-space representations of vocalizations using UMAP


This tutorial contains a sequence of jupyter notebook files that help you generate latent space representations from input audio files, evaluate them and generate an interactive visualization.

<p align="center">
  <img src="/example_imgs/tool_image.png" width="550" height="300" />
</p>

## 1. Structure

Keep the directory structure the way it is and put your data in the 'audio' and 'data' folder. Do not change the folder structure or location of notebooks or function files!

    ├── notebooks                              <- contains analysis scripts
    │   ├── 01_generate_spectrograms.ipynb      
    │   ├── ...           
    │   └── ...        
    ├── audio                                  <- ! put your input soundfiles in this folder !
    │   ├── call_1.wav     
    │   ├── call_2.wav         
    │   └── ...            
    ├── functions                              <- contains functions that will be called in analysis scripts
    │   ├── audio_functions.py            
    │   ├── ...                
    │   └── ...    
    ├── data                                   <- ! put a .csv metadata file of your input in this folder !
    │   └── info_file.csv                     
    ├── parameters                             
    │   └── spec_params.py                     <- this file contains parameters for spectrogramming (fft_win, fft_hop...)
    ├── environments    
    │   └── umap_tut_env.yaml                  <- conda environment file (linux)
    ├── ... 
    
    
## 2. Requirements

### 2.1. Packages, installations etc.

Python>=3.8. is recommended. I would recommend to __install the packages manually__, but a conda environment file is also included in /environments (created on Linux! Dependencies may differ for other OS!).

For manual install, these are the core packages:

>umap-learn

>librosa

>ipywidgets

>pandas=1.2.4

>seaborn

>pysoundfile=0.10.3

>voila

>hdbscan

>plotly

>graphviz

>networkx

>pygraphviz


Make sure to enable jupyter widgets with:
>jupyter nbextension enable --py widgetsnbextension


__NOTE__: Graphviz, networkx and pygraphviz are only required for one plot, so if you fail to install them, you can still run 99 % of the code.


#### This is an example for a manual installation on Windows with Python 3.8. and conda:

If you haven't worked with Python and/or conda (a package manager), an easy way to get started is to install anaconda or miniconda (only the basic/core parts of anaconda) first:

- Anaconda: [https://www.anaconda.com/products/individual-d](https://www.anaconda.com/products/individual-d)

- Miniconda: [https://docs.conda.io/en/latest/miniconda.html](https://docs.conda.io/en/latest/miniconda.html)

After successful installation, create and activate your environment with conda:

```
conda create --name my_env
conda activate my_env
```

Then, install the required core packages:

```
conda install -c conda-forge umap-learn
conda install -c conda-forge librosa
conda install ipywidgets
conda install pandas=1.2.4
conda install seaborn
conda install -c conda-forge pysoundfile=0.10.3
conda install -c conda-forge voila
conda install -c anaconda graphviz
conda install -c conda-forge hdbscan
conda install -c plotly plotly
conda install networkx
conda install -c conda-forge pygraphviz
```

Finally, enable ipywidgets in jupyter notebook

```
jupyter nbextension enable --py widgetsnbextension
```

Clone this repository or download as zip and unpack. Make sure to have the same structure of subdirectories as described in section "Structure" and prepare your input files as described in section "Input requirements".


Start jupyter notebook with
```
jupyter notebook
```

and select the first jupyter notebook file to start your analysis (see section "Where to start").


### 2.2. Input requirements

#### 2.2.1. Audio files

Use the provided example data (LINK) or put your own dataset of sound files in the subfolder "/audio" (make sure that the /audio folder contains __only__ your input files, nothing else).  
Each sound file should contain a single vocalization or syllable.
(You may have to detect and extract such vocal elements first, if working with acoustic recordings.)

Ideally, start and end of the sound file correspond exactly to start and end of the vocalization. 
If there are delays in the onset of the vocalizations, these should be the same for all sound files. 
Otherwise, vocalizations may appear dissimilar or distant in latent space simply because their onset times are different. 
If it is not possible to mark the start times correctly, use the timeshift option to generate UMAP embeddings,
but note that it comes at the cost of increased computation time.

#### 2.2.2. [Optional: Info file]

Use the .csv file provided with the example data (LINK) or, if you are using your own data,´ add a ";"-separated .csv file with headers containing the filenames of the input audio, some labels and any other additional metadata (if available) in the subfolder "/data". 
If some or all labels are unknown, there should still be a label column and unkown labels should be marked with "unknown".

Structure of info_file.csv:

    | filename   | label   | ...    |  .... 
    -----------------------------------------
    | call_1.wav | alarm   |  ...   |  ....   
    | call_2.wav | contact |  ...   |  ....  
    | ...        |  ...    |  ...   |  ....   

If you don't provide an info_file.csv, a default one will be generated, containing ALL files that are found in /audio and with all vocalizations labelled as "unknown".


## 3. Where to start

1. Start with 01_generate_spectrograms.ipynb to generate spectrograms from input audio files.
2. Generate latent space representations with 02a_generate_UMAP_basic.ipynb OR 02b_generate_UMAP_timeshift.ipynb 

3. You can now 
- __Evaluate__ the latent space representation with 03_UMAP_eval.ipynb,
 
- __Visualize__ the latent space representation by running 03_UMAP_viz_part_1_prep.ipynb and 03_UMAP_viz_part_2_tool.ipynb or

- __Apply clustering__ on the latent space representation with 03_UMAP_clustering.ipynb 


## 4. Data accessibility

This data is part of an ongoing study and is protected by copyright law, meaning that no one may reproduce, distribute, or create derivative works from it. If you want to access data, please get in touch.

