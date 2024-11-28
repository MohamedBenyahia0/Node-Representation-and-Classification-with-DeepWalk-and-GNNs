"""
Deep Learning on Graphs - ALTEGRAD - Nov 2024
"""

import networkx as nx
import numpy as np
from scipy.sparse.linalg import eigs
from scipy.sparse import diags, eye

from sklearn.linear_model import LogisticRegression
from sklearn.manifold import SpectralEmbedding
from sklearn.metrics import accuracy_score
from deepwalk import deepwalk



# Loads the karate network
G = nx.read_weighted_edgelist('../data/karate.edgelist', delimiter=' ', nodetype=int, create_using=nx.Graph())
print("Number of nodes:", G.number_of_nodes())
print("Number of edges:", G.number_of_edges())

n = G.number_of_nodes()

# Loads the class labels
class_labels = np.loadtxt('../data/karate_labels.txt', delimiter=',', dtype=np.int32)
idx_to_class_label = dict()
for i in range(class_labels.shape[0]):
    idx_to_class_label[class_labels[i,0]] = class_labels[i,1]

y = list()
for node in G.nodes():
    y.append(idx_to_class_label[node])

y = np.array(y)


############## Task 5
# Visualizes the karate network

##################
import matplotlib.pyplot as plt
nx.draw(G,pos=nx.spring_layout(G),node_color=y)
plt.show()
##################


############## Task 6
# Extracts a set of random walks from the karate network and feeds them to the Skipgram model
n_dim = 128
n_walks = 10
walk_length = 20
model = deepwalk(G, n_walks, walk_length, n_dim)

embeddings = np.zeros((n, n_dim))
for i, node in enumerate(G.nodes()):
    embeddings[i,:] = model.wv[str(node)]

idx = np.random.RandomState(seed=42).permutation(n)
idx_train = idx[:int(0.8*n)]
idx_test = idx[int(0.8*n):]

X_train = embeddings[idx_train,:]
X_test = embeddings[idx_test,:]

y_train = y[idx_train]
y_test = y[idx_test]


############## Task 7
# Trains a logistic regression classifier and use it to make predictions


##################
LogReg=LogisticRegression()
LogReg.fit(X_train,y_train)
y_pred=LogReg.predict(X_test)
print(f"DeepWalk Embedding Accuracy Score : {accuracy_score(y_test,y_pred)}")
##################


############## Task 8
# Generates spectral embeddings

##################
from sklearn.cluster import KMeans

def spectral_embedding(G, k=2):
    A = nx.adjacency_matrix(G) 
    n= A.shape[0]
    Dinv = diags(1/np.sum(A , axis=0))   
    L = eye(n)- Dinv.dot(A)
    S,U = eigs(L,k=k, which="SM")
    U=U.real
    return U
    
    
U=spectral_embedding(G, k=2)
Spec_train=U[idx_train,:]
Spec_test=U[idx_test,:]
y_train = y[idx_train]
y_test = y[idx_test]
LogReg=LogisticRegression()
LogReg.fit(Spec_train,y_train)
y_pred=LogReg.predict(Spec_test)
print(f"Spectral Embedding Accuracy Score : {accuracy_score(y_test,y_pred)}")

##################
