from import_modules import *

style.use('fivethirtyeight') # for matplotlib

all_categories = get_all_categories()


descriptor_matrix_10_10 = get_descriptor_matrix_10_10(all_categories)

# Write a variable into the pickle file:

with open('descriptor_matrix_10_10.pickle', 'wb') as handle:
    pickle.dump(descriptor_matrix_10_10,
                handle, protocol=pickle.HIGHEST_PROTOCOL)

# Read from the pickle file and assign it to a variable

# with open('descriptor_matrix_10_10.pickle', 'rb') as handle:
#     descriptor_matrix_10_10 = pickle.load(handle)
#     print(len(descriptor_matrix_10_10))


# descriptor_matrix_10_10 = get_descriptor_matrix_10_10(kmeans=KMeans(
#     n_clusters=100, random_state=0).fit(np.array(descriptor_matrix_10_10)))

# print("kmeans labels")
# print(kmeans.labels_)

# kmeans.predict()

# print("cluster centers")
# print(kmeans.cluster_centers_)

# If you get an error regarding cv2.xfeatures2d it's because in the new version that algorithm isn't free
# Do pip install opencv-contrib-python==3.4.1.15 to get rid of the error
