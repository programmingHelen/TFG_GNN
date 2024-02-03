# coding: utf-8
import numpy as np
from torch.utils import data
from tqdm import tqdm
import torch
import random
import scipy.sparse as sp


# Function to one-hot encode labels
def encode_onehot(labels):
    # Identify unique classes in labels
    classes = set(labels)
    
    # Create a dictionary mapping each class to its one-hot encoded vector
    classes_dict = {c: np.identity(len(classes))[i, :] for i, c in enumerate(classes)}
    
    # Apply one-hot encoding to labels
    labels_onehot = np.array(list(map(classes_dict.get, labels)), dtype=np.int32)
    
    return classes_dict, labels_onehot

# Function to perform row-normalization on a sparse matrix
def normalize(mx):
    # Calculate the sum of each row in the input matrix (mx)
    rowsum = np.array(mx.sum(1))  # Normalize each feature
    epsilon = 1e-10  # A small constant to avoid division by zero or very small values
    
    # Compute the inverse of the row sums, flatten the result
    r_inv = np.power(rowsum + epsilon, -1).flatten()
    r_inv[np.isinf(r_inv)] = 0.  # Handle infinite values by setting them to 0
    
    # Create a diagonal matrix from the inverse row sums
    r_mat_inv = sp.diags(r_inv)
    
    # Perform row-normalization by left-multiplying the input matrix with the diagonal matrix
    mx = r_mat_inv.dot(mx)
    
    return mx

# Function to convert a scipy sparse matrix to a torch sparse tensor
def sparse_mx_to_torch_sparse_tensor(sparse_mx):
    # Convert the sparse matrix to COOrdinate format and cast to float32
    sparse_mx = sparse_mx.tocoo().astype(np.float32)
    
    # Extract row indices, column indices, and data values
    indices = torch.from_numpy(np.vstack((sparse_mx.row, sparse_mx.col)).astype(np.int64))
    values = torch.from_numpy(sparse_mx.data)
    
    # Create a torch sparse tensor with extracted information and original shape
    shape = torch.Size(sparse_mx.shape)
    
    return torch.sparse.FloatTensor(indices, values, shape)

def load_data(opt):
    # Print loading message
    print("Loading {} dataset...".format(opt.network))

    # Load paper descriptions from .content file
    idx_features_labels = np.genfromtxt("./data/{}.content".format(opt.network), dtype=np.dtype(str))

    # Extract features and one-hot encode labels
    features = sp.csr_matrix(idx_features_labels[:, 1:-1], dtype=np.float32)  # Features
    classes_dict, labels = encode_onehot(idx_features_labels[:, -1])  # One-hot encoded labels

    # Create a dictionary with class names as keys and numeric labels as values
    numeric_labels_dict = {class_name: label for label, class_name in enumerate(classes_dict)}

    # Create an index map for node IDs
    idx = np.array(idx_features_labels[:, 0], dtype=np.dtype(str))
    idx_map = {j: i for i, j in enumerate(idx)}

    # Replace values in the first column of idx_features_labels with their corresponding values from idx_map
    for i in range(idx_features_labels.shape[0]):
        old_id = idx_features_labels[i, 0]
        new_id = idx_map.get(old_id, old_id)
        idx_features_labels[i, 0] = new_id

    # Load citation graph from .cites file
    edges_unordered = np.genfromtxt("./data/{}.cites".format(opt.network), dtype=np.dtype(str))  # Use dtype=str to handle alphanumeric IDs

    # Filter out rows that contain non-numeric values
    edges_filtered = []
    for i in range(edges_unordered.shape[0]):
        row = edges_unordered[i, :]
        # Check if both IDs in the row are present in the index map and are digits
        if all(idx_map.get(old_id) is not None and old_id.isdigit() for old_id in row):
            edges_filtered.append(row)

    # Convert the filtered edges_unordered to a numpy array
    edges_filtered = np.array(edges_filtered, dtype=np.int32)

    # Convert the updated edges_filtered to a flattened array of mapped values
    mapped_values = list(map(lambda old_id: idx_map.get(str(old_id), old_id), edges_filtered.flatten()))

    # Convert the mapped values to an array, replacing None with a default value (e.g., -1)
    edges = np.array([idx_map.get(str(old_id), -1) for old_id in edges_filtered.flatten()], dtype=np.int32).reshape(edges_filtered.shape)
    
    # Create a sparse adjacency matrix
    adj = sp.coo_matrix((np.ones(edges.shape[0]), (edges[:, 0], edges[:, 1])), shape=(labels.shape[0], labels.shape[0]), dtype=np.float32)
    adj = adj + adj.T.multiply(adj.T > adj) - adj.multiply(adj.T > adj)
    
    # Normalize features and adjacency matrix
    features = normalize(features)
    adj = normalize(adj + sp.eye(adj.shape[0]))

    # Create a map for each label to corresponding node indices
    labels_map = {i: [] for i in range(labels.shape[1])}
    labels = np.where(labels)[1]

    # Populate labels_map with node indices for each label
    for i in range(labels.shape[0]):
        labels_map[labels[i]].append(i)

    # Shuffle the indices within each label category
    for ele in labels_map:
        random.shuffle(labels_map[ele])

    # Split indices into training, validation, and test sets based on specified rates
    idx_train = list()
    idx_val = list()
    idx_test = list()

    for ele in labels_map:
        idx_train.extend(labels_map[ele][0:int(opt.train_rate * labels.shape[0])])
        idx_val.extend(labels_map[ele][int(opt.train_rate * labels[0]):int((opt.val_rate) * labels.shape[0])])
        idx_test.extend(labels_map[ele][int((opt.val_rate) * labels.shape[0]):])

    # Convert features and adjacency matrix to PyTorch FloatTensors
    features = torch.FloatTensor(np.array(features.todense()))
    adj = torch.FloatTensor(np.array(adj.todense()))

    # Return the preprocessed data
    return adj, features, labels, idx_train, idx_val, idx_test, edges, numeric_labels_dict



class Dataload(data.Dataset):

    def __init__(self, labels, id):
        self.data = id
        self.labels = labels

    def __getitem__(self, index):
        return index, self.labels[index]

    def __len__(self):
        return self.data.__len__()
