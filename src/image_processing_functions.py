from import_modules import *

# The constant number of K, this is mainly used to determin the k of K-means but also regulate many things like the lenght of the codebook
k = 600

# This function returns a list of strings corosponding to the names of the folders in the 'object_categories' directionary i.e. the names of the categories avalible.
def get_all_categories():
  categories = []
  for category in listdir("./object_categories"):
    #if category != ".DS_Store":
    categories.append(category)
  return categories

# This function returns a pandas data frame filled with imported images based on a given number of categories and images. in our case we load 5 categories with a max number of images
# set to 100 yileding 304 images. In the data frame we store the name of the file, the original category, and the image itself. We also declare a column 'type', here we 
# give half the images the lables 'train' and the other half 'test'. 
def get_results_dataframe(all_categories, n_categories, max_n_images):
  df = pd.DataFrame(columns=['file_name', 'category', 'img_array', 'type', 'bag_of_words']) # 'bag_of_words category is declared here but not filled til later

  for category_i, category in enumerate(all_categories[0:n_categories]):
    img_names = [img for img in listdir("./object_categories/" + category)]
    img_names = img_names[:max_n_images + 1]
  
    for i, img_name in enumerate(img_names[1:len(img_names)]):
      if img_name != ".DS_Store":
        img = cv2.imread("object_categories/" + category + "/" + img_name, cv2.IMREAD_GRAYSCALE)

      if i <= (len(img_names)/2) - 1:
        df = df.append({
          'file_name': img_name,
          'category': category,
          'img_array': img,
          'type': 'train',
          'bag_of_words': None
        }, ignore_index=True)

      elif i > (len(img_names)/2) - 1:
        df = df.append({
          'file_name': img_name,
          'category': category,
          'img_array': img,
          'type': 'test',
          'bag_of_words': None
        }, ignore_index=True)
    print("category loaded nr. ", category_i, "of ", n_categories)
  
  return df

# This function resruns a 2D numpy array with the dimentions (total number of SIFT features in traning set X 128). The 128 is of course the lenght of one SIFT descriptor. The array from this 
# fucntion is later used to create the codebook throudg the k-means algorimt.
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

# This function returns a 2D matrix with the dimentions (total number of SIFT features in the data set X lenght of the codebook, or K). The matrix is used to calculate the 
# bag of words later.
def calculate_distance_matrix (img_features, codebook):
  distance_matrix = np.zeros((len(img_features), len(codebook)))
  for i, row in enumerate(distance_matrix):
    for j, col in enumerate(row):
      distance_matrix[i,j] = distance.euclidean(img_features[i], codebook[j])
  return distance_matrix

# Function that creates K length sparse vector that represents the a given images "bag of words" based on the closest distances in the distance matrix created above.
def create_bag_of_words(distance_matrix):
  sparse_vector = np.zeros(k)
  for i, descriptor in enumerate(distance_matrix):
    minimum = min(descriptor)
    for j, word in enumerate(descriptor):
      if word == minimum:
        sparse_vector[j] +=1
  return sparse_vector

# This function loops over the whole data frame calculating caliing functions above and filling the 'bag_of_words'
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

# This function returns the Bhattacharyya distance between two given histograms i.e. bags of words
def calculate_bhattacharyya_distance(hist_1, hist_2):
  sum_ = 0
  for i in range(k):
    sum_ += np.sqrt(hist_1[i] * hist_2[i])
  return np.sqrt(1-(1/np.sqrt(np.mean(hist_1) * np.mean(hist_2) * (k**2))) * sum_)
    
# This function measures Bhattacharyya between the bag of words of a query images and all other bags of words in the data set. First it returns a ordered list of the categories with
# the category of the image with the lowest Bhattacharyya distance as the first index and the second lowest as the second and so on. Secondly it retruns a variable 'rank', rank represents 
# witch index from the list of suggestions correspond to the actual category of the image.  
def get_category_ranks_for_bow(img_bow, correct_category, df):
  query_distances = [] 
  nr_of_cat = len(set([cat for cat in df['category']]))

  for index, row in df.iterrows():
    bow = df["bag_of_words"].iloc[index]
    category = df["category"].iloc[index]
    query_distances.append((calculate_bhattacharyya_distance(img_bow, bow), category))

  query_distances = np.array(query_distances, dtype=[("distance", "<U32"), ("category", "<U32")])

  predicted_categories = [] 
  count = 0 
  for m in range(nr_of_cat): 
    count += 1
    query_distances.sort(order="distance")
    temp_category = query_distances[1][1] 
    predicted_categories.append(temp_category) 
    if temp_category == correct_category: 
      rank = count
    query_distances_list = query_distances.tolist()
    print("shape of list is", len(query_distances_list))
    newlist = []
    no_matter = 0
    for i in query_distances_list: 
      if temp_category == i[1]:
        no_matter += 1
      else:
        newlist.append(i)
    query_distances = np.array(newlist, dtype=[("distance", "<U32"), ("category", "<U32")])

  return predicted_categories, rank

# This function returns a new version of our data set with two new colunms, namely 'suggested_categories' and 'rank'. The content of these categories are based on the calculations from the 
# function above
def get_category_suggestions(df):
  df['suggested_categories'] = 0
  df['rank'] = 0
  for i, row in df.iterrows():
    img_bow = df["bag_of_words"].iloc[i]
    correct_category = df["category"].iloc[i]
    suggested_categories, rank = get_category_ranks_for_bow(img_bow, correct_category, df)
    df["suggested_categories"].iloc[i] = suggested_categories
    df["rank"].iloc[i] = rank
  return df

# This function returns the mean of the reciprocal ranks from a given data set.
def calculate_mean_reciprocal_rank(df):
  reciprocal_ranks = []
  for i, row in df.iterrows():
    reciprocal_ranks.append(1/df['rank'].iloc[i])
  return np.mean(reciprocal_ranks)

# This fucntion returns a percentage based on how many times in a given data set our functions 'top-3' suggestion was correct.
def calculate_top_3(df):
  count = 0
  for i, row in df.iterrows():
    if df['rank'].iloc[i] < 4:
      count += 1
  return (count/len(df))*100
