from import_modules import *

style.use('fivethirtyeight') # for matplotlib

all_categories = get_all_categories()

descriptor_matrix_10_10 = get_descriptor_matrices(all_categories, 10 , 10)
#descriptor_matrix_all = get_descriptor_matrix(all_categories, len(all_categories), # needs new fuction for finding nr. of images)

#Write a variable into the pickle file:
with open('./pickles/descriptor_matrix_10_10.pickle', 'wb') as handle:
    pickle.dump(descriptor_matrix_10_10,
                handle, protocol=pickle.HIGHEST_PROTOCOL)

# Calculate K-Means 
descriptor_matrix_10_10 =  np.array(descriptor_matrix_10_10)
kmeans_model = KMeans(n_clusters=k).fit(descriptor_matrix_10_10) 

#saves the k-means model
pickle.dump(kmeans_model, open('./pickles/codebook.pickle', 'wb')) # saves the k-means model

# Load and calculate SIFT features for a random image for testing.
#img = cv2.imread("./object_categories/sunflower/image_0015.jpg", cv2.IMREAD_GRAYSCALE)
#img_descriptors = get_sift_descriptors_for_img(img)

# kmeans.predict()

# Declare codebook from K-Means algorithm centers
codebook = kmeans_model.cluster_centers_

# 
#distance_matrix = create_distance_matrix(codebook, img_descriptors)

with open('./pickles/temp_distance_matrix.pickle', 'wb') as handle:
    pickle.dump(distance_matrix,
                handle, protocol=pickle.HIGHEST_PROTOCOL)

df = create_bags_of_words(df)

with open("./pickles/data_frame.pickle", "wb") as handle:
    pickle.dump(df, handle, protocol=pickle.HIGHEST_PROTOCOL)