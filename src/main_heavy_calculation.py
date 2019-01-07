from import_modules import *

style.use('fivethirtyeight') # for matplotlib

all_categories = get_all_categories()

descriptor_matrix_10_10 = get_descriptor_matrix_10_10(all_categories)
descriptor_matrix_all = get_descriptor_matrix(all_categories)

#Write a variable into the pickle file:
with open('../pickles/descriptor_matrix_10_10.pickle', 'wb') as handle:
    pickle.dump(descriptor_matrix_10_10,
                handle, protocol=pickle.HIGHEST_PROTOCOL)

descriptor_matrix_10_10 =  np.array(descriptor_matrix_10_10)
kmeans_model = KMeans(n_clusters=400).fit(descriptor_matrix_10_10) 

#saves the k-means model
pickle.dump(kmeans_model, open('../pickles/codebook.pickle', 'wb')) # saves the k-means model

img = cv2.imread("../object_categories/sunflower/image_0015.jpg", cv2.IMREAD_GRAYSCALE)
img_descriptors = get_sift_descriptors_for_img(img)

# kmeans.predict()

codebook = kmeans_model.cluster_centers_


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



temp_distance_matrix = create_distance_matrix(codebook, img_descriptors)

with open('../pickles/temp_distance_matrix.pickle', 'wb') as handle:
    pickle.dump(temp_distance_matrix,
                handle, protocol=pickle.HIGHEST_PROTOCOL)