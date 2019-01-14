from import_modules import *

# List with names of categories
all_categories = get_all_categories()

# Datafram with 304 images over 5 categories.
df_5_categories_100_images = get_results_dataframe(
    all_categories, n_categories=5, max_n_images=100)

# Calculates SIFT features from the training set
sift_for_codebook = calculate_sift_features_for_codebook(df_5_categories_100_images)
# Runs a k-means algorithm on the SIFT features with K = 600
kmeans_model = MiniBatchKMeans(n_clusters=k).fit(sift_for_codebook) 
# The 'codebook' is a list of centroids from the k-means model above 
codebook = kmeans_model.cluster_centers_

# The line below loads bags of words and places them in the data frame
df_5_categories_100_images_bow = create_bags_of_words(df_5_categories_100_images, codebook)

# The line below calculates suggestions and ranks and loads it into the data frame
df_5_categories_100_images_bow_suggestions = get_category_suggestions(df_5_categories_100_images_bow)

# Split the data frame into the traning and test set.
df_bow_with_suggestions_train = df_5_categories_100_images_bow_suggestions.loc[(df_5_categories_100_images_bow_suggestions['type'] == 'train')].reset_index()
df_bow_with_suggestions_test = df_5_categories_100_images_bow_suggestions.loc[(df_5_categories_100_images_bow_suggestions['type'] == 'test')].reset_index()

print(calculate_mean_reciprocal_rank(df_bow_with_suggestions_train), calculate_top_3(df_bow_with_suggestions_train))
print(calculate_mean_reciprocal_rank(df_bow_with_suggestions_test), calculate_top_3(df_bow_with_suggestions_test))