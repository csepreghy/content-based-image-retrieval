from import_modules import *

# Read Data frame from .pickle file 
with open('pickles/df_20_categories_all_images_k500.pickle', 'rb') as handle:
    df = pickle.load(handle)
   # print(len(descriptor_matrix_10_10))
print("Data frame: ", df.head(), df.tail())

svm_model = create_svm_model(df)


# Read K-means model from .pickle file
# loads the k-means model
# kmeans_model = pickle.load(open('pickles/codebook.pickle', 'rb'))
# codebook = kmeans_model.cluster_centers_

# print("codebook dimentions: ", codebook.shape)

