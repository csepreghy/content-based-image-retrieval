#Load the pickled data frame iwth BoWs
with open('pickles/df_5_categories_10_images_bow.pickle', 'rb') as handle:
    df_bow = pickle.load(handle)
print("opened dataframe with BoWs")

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
# print("X_train made with shape", X_train.shape)
# print(X_train)

#Make y_train
def get_y_train_BoWs(dataframe):
    y_train_BoWs = []
    for i, row in dataframe.loc[dataframe['type'] == 'train'].iterrows():
        label = dataframe.at[i, 'category']
        y_train_BoWs.append(label)
        #print("added label", dataframe.at[i, 'category'], "to y_train_BoWs")
    y_train_BoWs = np.array(y_train_BoWs)
    return y_train_BoWs
# y_train = get_y_train_BoWs(df_bow)
# print("y_train made with shape", y_train.shape)

#Make X_test
def get_X_test_BoWs(dataframe):
    X_test_BoWs = []
    for i, row in dataframe.loc[dataframe['type'] == 'test'].iterrows():
        bow = dataframe.at[i, 'bag_of_words']
        X_test_BoWs.append(bow)
        #print("added bow", dataframe.at[i, 'bag_of_words'], "to X_test_BoWs")
    X_test_BoWs = np.array(X_test_BoWs)
    return X_test_BoWs
# X_test = get_X_test_BoWs(df_bow)
# print("X_test made with shape", X_test.shape)

#Make y_test
def get_y_test_BoWs(dataframe):
    y_test_BoWs = []
    for i, row in dataframe.loc[dataframe['type'] == 'test'].iterrows():
        label = dataframe.at[i, 'category']
        y_test_BoWs.append(label)
        #print("added label", dataframe.at[i, 'category'], "to y_test_BoWs")
    y_test_BoWs = np.array(y_test_BoWs)
    return y_test_BoWs
# y_test = get_y_test_BoWs(df_bow)
# print("y_test made with shape", y_test.shape)