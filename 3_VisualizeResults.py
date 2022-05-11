import itk
import napari
import os

import numpy as np
from skimage.transform import downscale_local_mean

# What does it do?
#       loads specified segmentation, matching img, and displays in napari

# set path to segmentation
# path_to_segmentation = 'results_cm_2d_stitch/170626_2_pred.tif'
path_to_segmentation = 'results_cm/171031_2_pred.tif'


path_to_image_data = 'data/images'
stackname = os.path.splitext(os.path.basename(path_to_segmentation))[0].replace('_pred', "")
path_to_matching_wga = os.path.join(path_to_image_data, f'deblur_{stackname}_c3.h5')

img = downscale_local_mean(itk.GetArrayFromImage(itk.imread(path_to_matching_wga)), (1, 2, 2)).astype(np.uint8)
seg = itk.imread(path_to_segmentation)

viewer = napari.view_image(img)
viewer.add_labels(seg)
napari.run()