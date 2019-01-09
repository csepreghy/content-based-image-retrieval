from import_modules import *

style.use('fivethirtyeight') # for matplotlib

all_categories = get_all_categories()

df = get_results_dataframe(all_categories, n_categories=len(all_categories))
print(df.head(30))
print(df.tail())

sift_for_codebook = calculate_sift_features_for_codebook(df)
print("sift for codebook: ",sift_for_codebook.shape)
print("sift_for_codebook_done")

with open("./pickles/dataframe.pickle", "wb") as handle:
    pickle.dump(df, handle, protocol=pickle.HIGHEST_PROTOCOL)

# Calculate K-Means 
kmeans_model = MiniBatchKMeans(n_clusters=k).fit(sift_for_codebook) 
print("K-means model done")
codebook = kmeans_model.cluster_centers_

# saves the k-means model
with open("./pickles/codebook.pickle", "wb") as handle:
    pickle.dump(codebook, handle, protocol=pickle.HIGHEST_PROTOCOL)

# This calculates the bag of words for each row in the data frame
# df = create_bags_of_words(df, codebook)
# with open("./pickels/dataframe.pickle", "wb") as handle:
#     pickle.dump(df, handle, protocol=pickle.HIGHEST_PROTOCOL)

print("Fin")
