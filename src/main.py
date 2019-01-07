from import_modules import *

style.use('fivethirtyeight') # for matplotlib

all_categories = get_all_categories()

get_results_dataframe(all_categories, n_categories=5)

descriptor_matrix_10 = get_descriptor_matrices(all_categories, 10, 10)


# Write a variable into the pickle file:

# with open('../pickles/descriptor_matrix_10_10.pickle', 'wb') as handle:
#     pickle.dump(descriptor_matrix_10_10,
#                 handle, protocol=pickle.HIGHEST_PROTOCOL)

# Read from the pickle file and assign it to a variable

with open('./pickles/descriptor_matrix_10_10.pickle', 'rb') as handle:
    descriptor_matrix_10_10 = pickle.load(handle)
    print(len(descriptor_matrix_10_10))

# descriptor_matrix_10_10 =  np.array(descriptor_matrix_10_10)
# kmeans_model = KMeans(n_clusters=400).fit(descriptor_matrix_10_10) 

#saves the k-means model
# pickle.dump(kmeans_model, open('../pickles/codebook.pickle', 'wb')) # saves the k-means model

kmeans_model = pickle.load(open('./pickles/codebook.pickle', 'rb'))  # loads the k-means model

img = cv2.imread("./object_categories/sunflower/image_0015.jpg", cv2.IMREAD_GRAYSCALE)
img_descriptors = get_sift_descriptors_for_img(img)
# print("len of img_descriptors")
# print(len(img_descriptors))

# print("cluster centers")
# print(loaded_kmeans_model.cluster_centers_)
# print("kmeans labels")
# print(len(loaded_kmeans_model.labels_))

# kmeans.predict()

codebook = kmeans_model.cluster_centers_

#print("cluster centers")
#print(codebook)
# If you get an error regarding cv2.xfeatures2d it's because in the new version that algorithm isn't free
# Do pip install opencv-contrib-python==3.4.1.15 to get rid of the error
# print("cluster centers")

def measure_eucledian_distance(vector1, vector2):
    sum = 0
    for i in range(128):
        sum += (vector1[i] - vector2[i])**2
    return np.sqrt(sum)

def create_distance_matrix(codebook, features):
    k = len(codebook) 
    distance_matrix = []
    index_descriptors = 0
    count = 0
    for descriptor in features:
        row = []
        index_codebook = 0
        for codeword in codebook:
            row.append(measure_eucledian_distance(codeword, descriptor))
            index_codebook += 1
        index_descriptors += 1
        distance_matrix.append(row)
        print(int(index_descriptors/len(features)*100), "%")
    return distance_matrix


def create_bag_of_words(distance_matrix):
    k = len(codebook)
    sparse_vector = np.zeros(k)
    for i in range(len(sparse_vector)):
        minimum =  min (distance_matrix[i])
        for j in range(len(distance_matrix[i])):
            if distance_matrix[i][j] == minimum:
                sparse_vector[j] += 1
    return sparse_vector

# temp_distance_matrix = create_distance_matrix(codebook, img_descriptors)

# with open('../pickles/temp_distance_matrix.pickle', 'wb') as handle:
#     pickle.dump(temp_distance_matrix,
#                 handle, protocol=pickle.HIGHEST_PROTOCOL)

with open('./pickles/temp_distance_matrix.pickle', 'rb') as handle:
    temp_distance_matrix = pickle.load(handle)

test_bag_of_words = create_bag_of_words(temp_distance_matrix) 
print("bag og words: ",test_bag_of_words)       
print("sum of bag of words: ", sum(test_bag_of_words))
