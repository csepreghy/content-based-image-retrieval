from import_modules import *

style.use('fivethirtyeight') # for matplotlib

all_categories = get_all_categories()

df = get_results_dataframe(all_categories, n_categories=len(all_categories))
print(df.head(30))
#print(df.tail())

sift_for_codebook = calculate_sift_features_for_codebook(df)
print("sift for codebook: ",sift_for_codebook.shape)
print("sift_for_codebook_done")
# Calculate K-Means 
kmeans_model = MiniBatchKMeans(n_clusters=k).fit(sift_for_codebook) 
print("K-means model done")
codebook = kmeans_model.cluster_centers_

saves the k-means model
with open("./pickles/codebook", "wb") as handle:
    pickle.dump(codebook, handle, protocol=pickle.HIGHEST_PROTOCOL)
print("Fin")
# kmeans.predict()

#df = create_bags_of_words(df, codebook)
#print(df)

# with open("./pickles/data_frame.pickle", "wb") as handle:
#     pickle.dump(df, handle, protocol=pickle.HIGHEST_PROTOCOL)
