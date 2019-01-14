from import_modules import *
from sklearn.feature_extraction.text import TfidfTransformer
from sklearn.datasets import make_classification
from sklearn import preprocessing

#Load the pickled data frame iwth BoWs
with open('./pickles/df_5_categories_10_images_bow-3.pickle', 'rb') as handle:
    df_bow = pickle.load(handle)
print("Opened dataframe with BoWs.")

#Make X_train
def get_X_train_BoWs(dataframe):
    X_train_BoWs = []
    for i, row in dataframe.loc[dataframe['type'] == 'train'].iterrows():
        bow = dataframe.at[i, 'bag_of_words']
        X_train_BoWs.append(bow)
        #print("added bow", dataframe.at[i, 'bag_of_words'], "to X_train_BoWs")
    X_train_BoWs = np.array(X_train_BoWs)
    return X_train_BoWs
X_train = get_X_train_BoWs(df_bow)
#print("X_train made with shape", X_train.shape)

#Make y_train
def get_y_train_BoWs(dataframe):
    y_train_BoWs = []
    encoder = LabelEncoder()
    for i, row in dataframe.loc[dataframe['type'] == 'train'].iterrows():
        label = dataframe.at[i, 'category']
        y_train_BoWs.append(label)
        #print("added label", dataframe.at[i, 'category'], "to y_train_BoWs")
    y_train_BoWs = np.array(y_train_BoWs)
    return y_train_BoWs
y_train = get_y_train_BoWs(df_bow)
#print("y_train made with shape", y_train.shape)

#Make X_test
def get_X_test_BoWs(dataframe):
    X_test_BoWs = []
    for i, row in dataframe.loc[dataframe['type'] == 'test'].iterrows():
        bow = dataframe.at[i, 'bag_of_words']
        X_test_BoWs.append(bow)
        #print("added bow", dataframe.at[i, 'bag_of_words'], "to X_test_BoWs")
    X_test_BoWs = np.array(X_test_BoWs)
    return X_test_BoWs
X_test = get_X_test_BoWs(df_bow)
#print("X_test made with shape", X_test.shape)

#Make y_test
def get_y_test_BoWs(dataframe):
    y_test_BoWs = []
    for i, row in dataframe.loc[dataframe['type'] == 'test'].iterrows():
        label = dataframe.at[i, 'category']
        y_test_BoWs.append(label)
        #print("added label", dataframe.at[i, 'category'], "to y_test_BoWs")
    y_test_BoWs = np.array(y_test_BoWs)
    return y_test_BoWs
y_test = get_y_test_BoWs(df_bow)
#print("y_test made with shape", y_test.shape)

# #Uncomment to use tf-idf instaed of simple BoWs
def get_tfidf(sparse_vectors):
    tfidf = TfidfTransformer(use_idf=False)
    tfidf_vector = tfidf.fit_transform(sparse_vectors)
    # tfidf_vector = np.array(tfidf_vector)
    return tfidf_vector
# # X_train = get_tfidf(X_train)
# # X_test = get_tfidf(X_test)

# Encode our labels
labels = y_train.tolist()
for i in y_test:
    labels.append(i)
# print(labels)
# print(len(labels))

le = preprocessing.LabelEncoder()
le.fit(labels)
#print(list(le.classes_))
y_train_encoded = le.transform(y_train)
y_test_encoded = le.transform(y_test)

# # To transform the encoded labels back use:
# # list(le.inverse_transform([2, 2, 1]))
# # ['tokyo', 'tokyo', 'paris']

# Creates linear SVC classifier with a model that can predict category from BoW
# The accuracy of the model is printed at the end. You can feed it with any dataframe
# that has different number of categories and images.
# Apply linear SVC

# Creates SVM classifier with a model that can predict category from BoW
# The accuracy of the model is printed at the end. You can feed it with any dataframe
# that has different number of categories and images.
def create_svm_model(dataframe):
  clf = svm.SVC(gamma=0.001)

  X_train = get_X_train_BoWs(dataframe)
  X_test = get_X_test_BoWs(dataframe)

  y_train = get_y_train_BoWs(dataframe)

  clf.fit(X_train, y_train)
  
  y_pred = clf.predict(X_test)
  y_test = get_y_test_BoWs(dataframe)

  print("SVM classification performed with a score of:", accuracy_score(y_test, y_pred))
create_svm_model(df_bow)
