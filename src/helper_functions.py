from import_modules import *

def get_all_categories():
  categories = [category for category in listdir("./object_categories")]
  return categories


#takes two histograms of the test image and the codebook and returns the similarity measure 
hist_1 = []
hist_2 = []
#sim_neasure = compareHist(hist_1, hist_2, cv.CV_COMP_BHATTACHARYYA)

#print(sim_neasure)

