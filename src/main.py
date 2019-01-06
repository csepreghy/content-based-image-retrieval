from import_modules import *

style.use('fivethirtyeight') # for matplotlib

all_categories = get_all_categories()


# descriptor_matrix_10_10 = get_descriptor_matrix_10_10(all_categories)

# Write a variable into the pickle file:

# with open('pickles/descriptor_matrix_10_10.pickle', 'wb') as handle:
#     pickle.dump(descriptor_matrix_10_10,
#                 handle, protocol=pickle.HIGHEST_PROTOCOL)

# Read from the pickle file and assign it to a variable

with open('pickles/descriptor_matrix_10_10.pickle', 'rb') as handle:
    descriptor_matrix_10_10 = pickle.load(handle)
    print(len(descriptor_matrix_10_10))

# kmeans = KMeans(n_clusters=200)
# kmeans_model = kmeans.fit(np.array(descriptor_matrix_10_10))

# saves the k-means model
# pickle.dump(kmeans_model, open('pickles/codebook.pickle', 'wb'))

loaded_kmeans_model = pickle.load(
    open('pickles/codebook.pickle', 'rb'))  # loads the k-means model

img = cv2.imread("object_categories/ant/image_0015.jpg", cv2.IMREAD_GRAYSCALE)
img_descriptors = get_sift_descriptors_for_img(img)
print("len of img_descriptors")
print(len(img_descriptors))

# kmeans.predict()

# print("cluster centers")
# print(loaded_kmeans_model.cluster_centers_)

# If you get an error regarding cv2.xfeatures2d it's because in the new version that algorithm isn't free
# Do pip install opencv-contrib-python==3.4.1.15 to get rid of the error
