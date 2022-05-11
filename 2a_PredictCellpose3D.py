import skimage.io
import numpy as np
import os
import skimage.io
import tqdm
import itk
from cellpose import models
from skimage.transform import downscale_local_mean
import pickle

# What does it do?
#       predicts cellpose in 2.5D mode


# first try on train images, later do test
with open('data/train_paths.pkl', 'rb') as f:
    path_to_segmentations = pickle.load(f)

path_to_image_data = 'data/images'

imgs = []
for path_to_segmentation in tqdm.tqdm(path_to_segmentations):
    stackname = os.path.splitext(os.path.basename(path_to_segmentation))[0]
    path_to_matching_wga = os.path.join(path_to_image_data, f'deblur_{stackname}_c3.h5')
    img = downscale_local_mean(itk.GetArrayFromImage(itk.imread(path_to_matching_wga)), (1, 2, 2)).astype(np.uint8)
    imgs.append(img)

nimg = len(imgs)

path_to_model = 'data/extracted_slices_train/models/' \
                'cellpose_residual_on_style_on_concatenation_off_extracted_slices_train_2022_05_11_13_24_15.258300'

save_dir = 'results_cm/'
if not os.path.exists(save_dir):
    os.makedirs(save_dir)

use_GPU = True
omni = False
diameter = None
channels = [0, 0]

model = models.CellposeModel(gpu=use_GPU, pretrained_model=path_to_model, net_avg=False)

# TODO: optimize
cellprob_threshold = 0
flow_threshold = 0.4

for img_idx, img in enumerate(tqdm.tqdm((imgs))):
    file_name = os.path.splitext(os.path.basename(path_to_segmentations[img_idx]))[0]
    mask, flow, style = model.eval(img, diameter=diameter,
                                   flow_threshold=flow_threshold,
                                   cellprob_threshold=cellprob_threshold,
                                   channels=channels,
                                   do_3D=True)

    mask_output_name = os.path.join(save_dir, file_name + "_pred.tif")
    mask = mask.astype(np.uint16)

    skimage.io.imsave(mask_output_name, mask, check_contrast=False)
