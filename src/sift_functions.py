import cv2
import cv2
import matplotlib.pyplot as plt
from matplotlib import style

style.use('fivethirtyeight')

def get_descriptor_matrix(N_IMAGES, img_category):
  sift = cv2.xfeatures2d.SIFT_create()
  descriptor_matrix = []

  print(N_IMAGES)
  for i in range(N_IMAGES):
    print(i)
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
      keypoints, descriptors = sift.detectAndCompute(img, None)
      img = cv2.drawKeypoints(img, keypoints, None)
      descriptor_matrix.append(descriptors)
    
    except:
      break

    # plt.imshow(img)
    # plt.show()
    # cv2.imshow("Image", img)
    # cv2.waitKey(0)
    # cv2.destroyAllWindows()

  return descriptor_matrix
