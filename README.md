Graph Neural Network Architectures
This repository provides an in-depth exploration and implementation of prominent Graph Neural Network (GNN) architectures, including Graph Attention Networks (GAT), GraphSAGE, and Graph Convolutional Networks (GCN). The implementations are accompanied by detailed explanations, code comments, and visual aids to facilitate understanding.

Table of Contents
Graph Attention Networks (GAT)

Overview: GAT, introduced by Velickovic et al., leverages attention mechanisms to enhance information aggregation in graph-structured data. The architecture's flexibility and power in handling varying graph sizes make it a prominent choice for tasks like node classification, link prediction, and graph classification.
Challenges: Despite its success, GAT faces challenges like over-smoothing, prompting the development of variants like GATv2.
Visual Aid: 
GraphSAGE

Overview: GraphSAGE is designed for inductive representation learning on large graphs, efficiently handling scalability issues. It incorporates neighbor sampling, aggregation functions, and layer-wise aggregation for applications like node classification, link prediction, and graph classification.
Applications: Used in real-world scenarios such as Pinterest's PinSAGE and UberEats for recommendations.
Visual Aid: 
Graph Convolutional Networks (GCN)

Overview: GCN, a natural choice for graph-structured data, performs message passing and aggregation to capture relational information. Its success in various applications like node classification and graph classification has made it a popular tool in domains such as social networks and bioinformatics.
Generalization: GCNs can be generalized in both spectral and spatial domains, inspiring subsequent models and applications.
Visual Aid: 
Data Preprocessing and Utilization
The repository includes a comprehensive data loading and preprocessing module, accommodating diverse datasets. The preprocessing addresses challenges encountered in large datasets, including parallelization to enhance computational efficiency.

Parallelized Workflow: The initial code, while effective for smaller datasets, faced computational challenges with larger datasets. To address this, we parallelized the workflow, breaking down the processing load into smaller, independent jobs to improve efficiency. Despite the reduction in processing time, larger datasets necessitated significant computing resources.

Dataset Information: The repository supports multiple datasets, including Cora, Citeseer, and a large RO dataset. The preprocessing involves tasks like one-hot encoding labels, normalizing sparse matrices, and handling edge information.

Code Structure
The implementations are structured for clarity and ease of understanding. Additionally, code comments and documentation have been included to provide detailed insights into each architecture's components and functions.

