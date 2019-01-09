from import_modules import *

style.use('fivethirtyeight') # for matplotlib

all_categories = get_all_categories()

df = get_results_dataframe(all_categories, n_categories=2)
print(df.head(30))
#print(df.tail())

# We will not use "desctiptor_matrix_10_10, " we will genereate the descriptors
# for the clustering by taking the image arrays of train images from the 
# data frame

# descriptor_matrix_train = calculate_sift_features_for_codebook(df)
#descriptor_matrix_train = get_descriptor_matrices(all_categories, , len(all_categories)

#Write a variable into the pickle file:
# with open('./pickles/descriptor_matrix_10_10.pickle', 'wb') as handle:
#     pickle.dump(descriptor_matrix_10_10,
#                 handle, protocol=pickle.HIGHEST_PROTOCOL)

# Calculate K-Means 
# descriptor_matrix_train =  np.array(descriptor_matrix_train)
# kmeans_model = KMeans(n_clusters=k).fit(descriptor_matrix_train) 

# This is a temporary measure
kmeans_model = pickle.load(open('pickles/codebook.pickle', 'rb'))  # loads the k-means model

# saves the k-means model
# pickle.dump(kmeans_model, open('./pickles/codebook.pickle', 'wb')) # saves the k-means model

# kmeans.predict()

# Declare codebook from K-Means algorithm centers
codebook = kmeans_model.cluster_centers_
#print("codebook info: ",type(codebook), codebook.shape, codebook)

df = create_bags_of_words(df, codebook)
print(df)

with open("./pickles/data_frame.pickle", "wb") as handle:
    pickle.dump(df, handle, protocol=pickle.HIGHEST_PROTOCOL)
