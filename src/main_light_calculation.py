from import_modules import *

# Read Data frame from .pickle file 
with open('pickles/dataframe.pickle', 'rb') as handle:
    df = pickle.load(handle)
   # print(len(descriptor_matrix_10_10))

# Read K-means model from .pickle file
kmeans_model = pickle.load(open('pickles/codebook.pickle', 'rb'))  # loads the k-means model
codebook = kmeans_model.cluster_centers_


