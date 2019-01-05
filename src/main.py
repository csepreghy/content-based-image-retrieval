from import_modules import *

style.use('fivethirtyeight') # for matplotlib

all_categories = get_all_categories()

# descriptor_matrix_all_categories = get_descriptor_matrix_all_categories(
#     all_categories)


# with open('descriptor_matrix_all_categories.pickle', 'wb') as handle:
#     pickle.dump(descriptor_matrix_all_categories,
#                 handle, protocol=pickle.HIGHEST_PROTOCOL)

with open('descriptor_matrix_all_categories.pickle', 'rb') as handle:
    descriptor_matrix_all_categories = pickle.load(handle)
    print(len(descriptor_matrix_all_categories[12]['descriptor_matrix']))

# descriptor_matrix = get_descriptor_matrix(N_IMAGES, img_category = "buddha")
# didn't make the img_category into a constant because we might have a list instead to loop through

# kmeans = KMeans(n_clusters=100, random_state=0).fit(np.array(descriptor_matrix))
# print(kmeans.labels_)

# kmeans.predict()

#print(kmeans.cluster_centers_)

# If you get an error regarding cv2.xfeatures2d it's because in the new version that algorithm isn't free
# Do pip install opencv-contrib-python==3.4.1.15 to get rid of the error
