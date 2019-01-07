# Python modules
import cv2
import matplotlib.pyplot as plt
from matplotlib import style
from os import listdir
from sklearn.cluster import KMeans
import numpy as np
import pickle
import pandas as pd

# Our own files
from constants import N_IMAGES
from image_processing_functions import *
from helper_functions import get_all_categories
