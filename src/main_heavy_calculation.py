from import_modules import *

style.use('fivethirtyeight') # for matplotlib

all_categories = get_all_categories()

#df_all_categories = get_results_dataframe(
#    all_categories, n_categories=len(all_categories), max_n_images=10)

df_5_categories_10_images = get_results_dataframe(
    all_categories, n_categories=5, max_n_images=10)

# print(df_5_categories_10_images.head(30))
# print(df_5_categories_10_images.tail())

sift_for_codebook = calculate_sift_features_for_codebook(df_5_categories_10_images)
# print("sift for codebook: ",sift_for_codebook.shape)
# print("sift_for_codebook_done")

# with open("./pickles/dataframe_all_categories.pickle", "wb") as handle:
#     pickle.dump(df_all_categories, handle, protocol=pickle.HIGHEST_PROTOCOL)
  
with open("./pickles/df_5_categories_10_images.pickle", "wb") as handle:
    pickle.dump(df_5_categories_10_images, handle, protocol=pickle.HIGHEST_PROTOCOL)

# Calculate K-Means 
kmeans_model = MiniBatchKMeans(n_clusters=k).fit(sift_for_codebook) 
# print("K-means model done")
codebook = kmeans_model.cluster_centers_

# saves the k-means model
with open("./pickles/codebook_5_categories_10_images.pickle", "wb") as handle:
    pickle.dump(codebook, handle, protocol=pickle.HIGHEST_PROTOCOL)

# This calculates the bag of words for each row in the data frame
df = create_bags_of_words(df_5_categories_10_images, codebook)

with open("./pickles/df_small_BoW.pickle", "wb") as handle:
    pickle.dump(df, handle, protocol=pickle.HIGHEST_PROTOCOL)

print("Fin")
