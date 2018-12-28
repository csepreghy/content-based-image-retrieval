from import_modules import *

style.use('fivethirtyeight') # for matplotlib

all_categories = get_all_categories()

descriptor_matrix = get_descriptor_matrix(N_IMAGES, img_category = "buddha")
# didn't make the img_category into a constant because we might have a list instead to loop through

kmeans = KMeans(n_clusters=100, random_state=0).fit(np.array(descriptor_matrix))
print(kmeans.labels_)

# kmeans.predict()

#print(kmeans.cluster_centers_)

# If you get an error regarding cv2.xfeatures2d it's because in the new version that algorithm isn't free
# Do pip install opencv-contrib-python==3.4.1.15 to get rid of the error
