from import_modules import *

# Read descriptor matrix from .pickle file 
with open('../pickles/descriptor_matrix_10_10.pickle', 'rb') as handle:
    descriptor_matrix_10_10 = pickle.load(handle)
    print(len(descriptor_matrix_10_10))

# Read K-means model from .pickle file
kmeans_model = pickle.load(open('../pickles/codebook.pickle', 'rb'))  # loads the k-means model

# Read distance matrix form .pickle file
with open('../pickles/temp_distance_matrix.pickle', 'rb') as handle:
    temp_distance_matrix = pickle.load(handle)

# Function that creates K lenght sparse vector that represents the a given images "bag of words"
def create_bag_of_words(distance_matrix):
    k = len(codebook)
    sparse_vector = np.zeros(k)
    for i in range(len(sparse_vector)):
        minimum =  min (distance_matrix[i])
        for j in range(len(distance_matrix[i])):
            if distance_matrix[i][j] == minimum:
                sparse_vector[j] += 1
    return sparse_vector


test_bag_of_words = create_bag_of_words(temp_distance_matrix) 

print("bag og words: ",test_bag_of_words)       
print("sum of bag of words: ", sum(test_bag_of_words))

