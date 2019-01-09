from import_modules import *

# Creates the data frame and fills all columns except for "bag_of_words"
def get_results_dataframe(all_categories, n_categories, max_n_images):
  df = pd.DataFrame(columns=['file_name', 'category', 'img_array', 'type', 'bag_of_words'])
  #sift = cv2.xfeatures2d.SIFT_create()
  #print(all_categories)

  for category_i, category in enumerate(all_categories[0:n_categories]):
    img_names = [img for img in listdir("./object_categories/" + category)]
    img_names = img_names[:max_n_images + 1]
    #if ".DS_Store" in img_names: img_names.remove(".DS_Store")
  
    for i, img_name in enumerate(img_names[1:len(img_names)]):
      if img_name != ".DS_Store":
        img = cv2.imread("object_categories/" + category + "/" + img_name, cv2.IMREAD_GRAYSCALE)

      if i <= len(img_names)/2:
        df = df.append({
          'file_name': img_name,
          'category': category,
          'img_array': img,
          'type': 'train',
          'bag_of_words': None
        }, ignore_index=True)

      elif i > len(img_names)/2:
        df = df.append({
          'file_name': img_name,
          'category': category,
          'img_array': img,
          'type': 'test',
          'bag_of_words': None
        }, ignore_index=True)
    print("category loaded nr. ", category_i, "of ", n_categories)
  
  return df

# Collects all sift features for training images in one np.array to use for clustering
def calculate_sift_features_for_codebook(df):
  sift = cv2.xfeatures2d.SIFT_create()
  df_only_train = df.loc[df['type'] == 'train']
  sift_features = []
  for i, row in df_only_train.iterrows():
    img = df.at[i, 'img_array']
    (keypoints, descriptors) = sift.detectAndCompute(img, None)
    if descriptors is not None:
      for descriptor in descriptors:
        if descriptor is not None:
          sift_features.append(descriptor)
        else: 
          print("empty descriptor found in row: ", i)
  sift_features = np.array(sift_features)
  print("features added for nr. of img: ", i, "of: ", len(df))
  return sift_features

# Creates distance matrix between image features and codewords in codebook
def calculate_distance_matrix (img_features, codebook):
  distance_matrix = np.zeros((len(img_features), len(codebook)))
  for i, row in enumerate(distance_matrix):
    for j, col in enumerate(row):
      distance_matrix[i,j] = distance.euclidean(img_features[i], codebook[j])
  return distance_matrix

# Function that creates K length sparse vector that represents the a given images "bag of words"
def create_bag_of_words(distance_matrix):
  sparse_vector = np.zeros(k)
  for i, descriptor in enumerate(distance_matrix):
    minimum = min(descriptor)
    for j, word in enumerate(descriptor):
      if word == minimum:
        sparse_vector[j] +=1
  #print("create bag of words: BoW done", )
  return sparse_vector

# Function to calculate each cell of the "bag of words" column
def create_bags_of_words(df, codebook):
  sift = cv2.xfeatures2d.SIFT_create()
  for index, row in df.iterrows():
    img = df["img_array"].iloc[index]
    (keypoints, descriptors) = sift.detectAndCompute(img, None)
    img_descriptors = descriptors
    distance_matrix = calculate_distance_matrix(img_descriptors, codebook)
    df["bag_of_words"].iloc[index] = create_bag_of_words(distance_matrix)
    print("Full iterations done: ", index)

  return df

def calculate_indian_distance(hist_1, hist_2):
  sum_ = 0
  for i in range(k):
    sum_ += np.sqrt(hist_1[i] * hist_2[i])
  return 1-(1/np.sqrt(np.mean(hist_1) * np.mean(hist_2) * (k**2))) * sum_


def calculate_hist_dist(test_BoW, df):
  hist_dist = []
  for i, row in df.iterrows():
    hist_2 = df["bag_of_words"].iloc[i]
    category = (df["category"].iloc[i])
    hist_dist.append((calculate_indian_distance(test_BoW, hist_2), category))
  # Mikkel you picese of shit, finish this function
  return 