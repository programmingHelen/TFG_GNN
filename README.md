# Graph Neural Network Architectures
This repository delves into the exploration and implementation of prominent Graph Neural Network (GNN) architectures, featuring Graph Attention Networks (GAT), GraphSAGE, and Graph Convolutional Networks (GCN). The project base was the repository (https://github.com/MrLeeeee/GCN-GAT-and-Graphsage/tree/master).

## Data folder
**Overview:**
* Contains all the datasets used for the simulations and the raw JSON files from Deezer.
* Dataset.py with all the preprocessing functions
* Deezer reader.ipynb auxiliary notebook to convert files from JSON and csv to .content and .cites

## Models folder
Includes the 3 architecture files in .py extension. With their spefic layers
BasicModel.py
Graphsage.py
PyGAT.py
PyGCN.py
__init__.py

## Utilization
1. Check the requirements.txt for the libraries versions.
2. Run the main.ipynb selecting the dataset and models you want.
