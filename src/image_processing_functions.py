from import_modules import *

def get_descriptor_matrix(N_IMAGES, img_category):
  sift = cv2.xfeatures2d.SIFT_create()
  descriptor_matrix = []

  #print(N_IMAGES)
  for i in range(N_IMAGES):
    #print(img_category, i)
    if i < 9:
      img = cv2.imread("object_categories/" + img_category + "/image_000" +
                       str(i + 1) + ".jpg", cv2.IMREAD_GRAYSCALE)

    elif i >= 9 & i < 99:
      img = cv2.imread("object_categories/" + img_category + "/image_00" +
                       str(i + 1) + ".jpg", cv2.IMREAD_GRAYSCALE)

    elif i >= 99 & i < 999:
      img = cv2.imread("object_categories/" + img_category + "/image_0" +
                       str(i + 1) + ".jpg", cv2.IMREAD_GRAYSCALE)

    try:
      (keypoints, descriptors) = sift.detectAndCompute(img, None)
      img = cv2.drawKeypoints(img, keypoints, None)
      
      descriptor_matrix += [descriptor for descriptor in descriptors]

    except:
      break

    # plt.imshow(img)
    # plt.show()
    # cv2.imshow("Image", img)
    # cv2.waitKey(0)
    # cv2.destroyAllWindows()

  return descriptor_matrix


def get_descriptor_matrices(all_categories, n_images, n_categories):
  descriptor_matrix = []
  for category in all_categories[0:n_categories]:
    descriptor_matrix = get_descriptor_matrix(
        N_IMAGES=n_images, img_category=category)
    descriptor_matrix += [
        descriptor for descriptor in descriptor_matrix]

  return descriptor_matrix

# def calculate_sift_features_for_codebook(df):
#   sift = cv2.xfeatures2d.SIFT_create()
#   sift_features = []
#   df_only_train = df[df["type"] == "train"]
#   for i, row in df.iterrows():
#     img = df_only_train[df_only_train["img_array"]].iloc[i]
#     (keypoints, descriptors) = sift.detectAndCompute(img, None)
#     sift_features.append(descriptors)
#   return sift_features

def measure_eucledian_distance(vector1, vector2):
  sum = 0
  for i in range(128):
    sum += (vector1[i] - vector2[i])**2
  return np.sqrt(sum)

def get_sift_descriptors_for_img(img):
  sift = cv2.xfeatures2d.SIFT_create()
  (keypoints, descriptors) = sift.detectAndCompute(img, None)
  return descriptors


def get_results_dataframe(all_categories, n_categories):
  df = pd.DataFrame(columns=['file_name', 'category', 'img_array', 'type', 'bag_of_words'])

  for category_i, category in enumerate(all_categories[0:n_categories]):
    #print("DataFrame creation: category " + str(category_i + 1) + "/" + str(n_categories))
    img_names = [img for img in listdir("./object_categories/" + category)]
    if ".DS_Store" in img_names: img_names.remove(".DS_Store")
  
    for i, img_name in enumerate(img_names):
      img = cv2.imread("object_categories/" + category + "/" + img_name, cv2.IMREAD_GRAYSCALE)
      
      if i < len(img_names)/2:
        df = df.append({
          'file_name': img_name,
          'category': category,
          'img_array': img,
          'type': 'train',
          'bag_of_words': None
        }, ignore_index=True)

      elif i >= len(img_names)/2:
        df = df.append({
          'file_name': img_name,
          'category': category,
          'img_array': img,
          'type': 'test',
          'bag_of_words': None
        }, ignore_index=True)
  
  return df


# Function that creates K lenght sparse vector that represents the a given images "bag of words"
def create_bag_of_words(distance_matrix):
  sparse_vector = np.zeros(k)
  for i in range(len(sparse_vector)):
    minimum =  min (distance_matrix[i])
    for j in range(len(distance_matrix[i])):
      if distance_matrix[i][j] == minimum:
          sparse_vector[j] += 1
  return sparse_vector

def calculate_distance_matrix (img_features, codebook):
  distance_matrix = np.zeros((len(img_features), len(codebook)))
  for i, row in enumerate(distance_matrix):
    for j, col in enumerate(row):
      distance_matrix[i,j] = distance.euclidean(img_features[i], codebook[j])
  return distance_matrix

# Function to calculate each cell of the "bag of words" column
def create_bags_of_words(df, codebook):
  #df = df[df['type'] == 'train']
  print(df.head())
  #print(df.head())
  for index, row in df.iterrows():
    #print("image array: ", df["img_array"].iloc[index])
    img_descriptors = get_sift_descriptors_for_img(df["img_array"].iloc[index])
    distance_matrix = calculate_distance_matrix(img_descriptors, codebook)
    print("created distance matrix for row: ", index)
    df["bag_of_words"].iloc[index] = create_bag_of_words(distance_matrix)
    print("created bag of words, test sum: ", sum(df["bag_of_words"].iloc[index]))
  return df
