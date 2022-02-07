# Geometrc Parameters for precise detection of Text Regions in MSER regions
AREA_LIM = 2.0e-5
PERIMETER_LIM = 1e-6
ASPECT_RATIO_LIM = 5.0
OCCUPATION_LIM = (0.23, 0.90)
COMPACTNESS_LIM = (3e-3, 1e-1)
#NMS
NMS_OVERLAP_THRESHOLD= 0.3

#MODEL PARAMS
MODEL_PATH= './weights.hdf5'
