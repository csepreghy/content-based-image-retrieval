from import_modules import *

style.use('fivethirtyeight') # for matplotlib

# all_categories = get_all_categories()


# descriptor_matrix_10_10 = get_descriptor_matrix_10_10(all_categories)

# # Write a variable into the pickle file:

# with open('descriptor_matrix_10_10.pickle', 'wb') as handle:
#     pickle.dump(descriptor_matrix_10_10,
#                 handle, protocol=pickle.HIGHEST_PROTOCOL)

# # Read from the pickle file and assign it to a variable

with open('descriptor_matrix_10_10.pickle', 'rb') as handle:
    descriptor_matrix_10_10 = pickle.load(handle)
    #print(len(descriptor_matrix_10_10))

# kmeans = KMeans(n_clusters=200)
# kmeans_model = kmeans.fit(np.array(descriptor_matrix_10_10))

# pickle.dump(kmeans_model, open('codebook.pickle', 'wb'))  # saves the k-means model

loaded_kmeans_model = pickle.load(
    open('codebook.pickle', 'rb'))  # loads the k-means model

# print("kmeans labels")
# print(len(loaded_kmeans_model.labels_))

# kmeans.predict()

codebook = loaded_kmeans_model.cluster_centers_

#print("cluster centers")
#print(codebook)
# If you get an error regarding cv2.xfeatures2d it's because in the new version that algorithm isn't free
# Do pip install opencv-contrib-python==3.4.1.15 to get rid of the error
def measure_eucledian_distance(vector1, vector2):
    sum = 0
    for i in range(128):
        sum += (vector1[i] - vector2[i])**2
    return np.sqrt(sum)

def create_distance_matrix(codebook, features):
    k = len(codebook) # should match the k value from the k-means algorithm 
    #distance_matrix = np.array((len(features), k))
    distance_matrix = []
    index_descriptors = 0
    for descriptor in features:
        row = []
        index_codebook = 0
        for codeword in codebook:
            #distance_matrix[index_descriptors][index_codebook].append(measure_eucledian_distance(codeword, descriptor))
            row.append(measure_eucledian_distance(codeword, descriptor))
            index_codebook += 1
        index_descriptors += 1
        distance_matrix.append(row)
        print("%", int(int(index_descriptors/len(features)*100)*0.3)*"=", int(100-int(index_descriptors/len(features)*20))*".", "%")
        print()
    return distance_matrix

def create_bag_of_words(distance_matrix):
    k = len(codebook)
    #sparse_vector = np.zeros(k)
    sparse_vector = []
    for i in distance_matrix:
        minimum = min(i)
        for j in i:
            count = 0
            if j == minimum: # and if minimum > "threshold"
                count += 1
        sparse_vector.append(count)
    print(sparse_vector)
        

test_bag_of_words = create_bag_of_words(create_distance_matrix(codebook, codebook)) 
print(test_bag_of_words)       