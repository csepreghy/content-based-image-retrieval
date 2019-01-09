from import_modules import *

# Read Data frame from .pickle file 
<<<<<<< HEAD
# with open('pickles/dataframe.pickle', 'rb') as handle:
#     df = pickle.load(handle)
=======
with open('pickles/df_5_categories_10_images_k800.pickle', 'rb') as handle:
    df = pickle.load(handle)
>>>>>>> f51ad794055ba53cc4dd80e0c3ca0f760c1016a9
   # print(len(descriptor_matrix_10_10))
#print("Data frame: ", df.head(), df.tail())

<<<<<<< HEAD
# with open('pickles/df_5_categories_10_images.pickle', 'rb') as handle:
#     df_small = pickle.load(handle)


# Read K-means model from .pickle file
codebook = pickle.load(open('pickles/codebook_5_categories_10_images.pickle', 'rb'))  # loads the k-means model
print('codebook dimentions: ', codebook.shape)
# df_small_BoW = create_bags_of_words(df_small, codebook)

# with open("./pickles/df_small_BoW.pickle", "wb") as handle:
#      pickle.dump(df_small_BoW, handle, protocol=pickle.HIGHEST_PROTOCOL)
=======
svm_model = create_svm_model(df)


# Read K-means model from .pickle file
# loads the k-means model
# kmeans_model = pickle.load(open('pickles/codebook.pickle', 'rb'))
# codebook = kmeans_model.cluster_centers_

# print("codebook dimentions: ", codebook.shape)
>>>>>>> f51ad794055ba53cc4dd80e0c3ca0f760c1016a9

df_small_BoW = pickle.load(open('./pickles/df_small_BoW.pickle', 'rb'))
print(df_small_BoW)
img_nr = 19
test_img = df_small_BoW['bag_of_words'].iloc[img_nr]
test = calculate_hist_dist(test_img, df_small_BoW)
print("closest: ", min(test),  "top 3: ", test, "real category: ", df_small_BoW['category'].iloc[img_nr])
