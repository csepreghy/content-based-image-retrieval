from import_modules import *

def get_descriptor_matrix(N_IMAGES, img_category):
  sift = cv2.xfeatures2d.SIFT_create()
  descriptor_matrix = []

  print(N_IMAGES)
  for i in range(N_IMAGES):
    print(img_category, i)
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


def get_sift_descriptors_for_img(img):
  sift = cv2.xfeatures2d.SIFT_create()
  (keypoints, descriptors) = sift.detectAndCompute(img, None)
  return descriptors


def get_results_dataframe(all_categories, n_categories):
  df = pd.DataFrame(columns=['file_name', 'category', 'img_array', 'type', 'bag_of_words'])

  for category in all_categories[0:n_categories]:
    img_names = [img for img in listdir("./object_categories/" + category)]
  
    for i, img_name in enumerate(img_names):
      img = cv2.imread("object_categories/" + category + "/" + img_name, cv2.IMREAD_GRAYSCALE)
      
      if i < len(img_names)/2:
        print("first if")
        df = df.append({
          'file_name': img_name,
          'category': category,
          'img_array': img,
          'type': 'train',
          'bag_of_words': None
        }, ignore_index=True)

      elif i >= len(img_names)/2:
        print("second if")
        df = df.append({
          'file_name': img_name,
          'category': category,
          'img_array': img,
          'type': 'test',
          'bag_of_words': None
        }, ignore_index=True)
  
  return df

