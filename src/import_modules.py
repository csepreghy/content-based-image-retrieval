# Python modules
import cv2
import matplotlib.pyplot as plt
from matplotlib import style
from os import listdir
from sklearn.cluster import KMeans, MiniBatchKMeans
from sklearn import svm
import numpy as np
import pickle
import pandas as pd
from scipy.spatial import distance
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import accuracy_score

# Our own files
from constants import N_IMAGES, k
from image_processing_functions import *
from helper_functions import get_all_categories
from ml_functions import *
