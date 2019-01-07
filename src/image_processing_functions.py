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


def get_descriptor_matrix_10_10(all_categories):
  descriptor_matrix_10_10 = []

  for category in all_categories[0:9]:
    descriptor_matrix = get_descriptor_matrix(N_IMAGES = 10, img_category=category)
    descriptor_matrix_10_10 += [
        descriptor for descriptor in descriptor_matrix]

  return descriptor_matrix_10_10


def get_sift_descriptors_for_img(img):
  sift = cv2.xfeatures2d.SIFT_create()
  (keypoints, descriptors) = sift.detectAndCompute(img, None)
  return descriptors
