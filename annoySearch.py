import pickle
from sklearn.neighbors import BallTree

vectors = pickle.load(open('./model/data/vectors.pkl', 'rb'))

k = 7  # Number of nearest neighbors to find
tree = BallTree(vectors, metric='minkowski')  # Use 'minkowski' metric for Euclidean distance

# Perform a k-nearest neighbor search for the first item
distances, indices = tree.query(vectors[:1], k=k)

# Indices contain the indices of the nearest neighbors
print("Nearest neighbors:", indices)

# Distances contain the distances to the nearest neighbors
print("Distances:", distances)
