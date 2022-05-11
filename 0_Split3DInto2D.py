import glob
import os
from random import shuffle

import itk
import numpy as np
from skimage.transform import downscale_local_mean
import tqdm
from skimage.measure import block_reduce
import pickle

# What does it do?
#       loads image and data, downsamples them, extract 2D tifs of XY, XZ, YZ

### PARAMETERS ###
path_to_segmentations = 'data/masks'
path_to_image_data = 'data/images'

downsample_factor = (1, 2, 2)
number_of_random_xy_cuts = 5
number_of_random_xz_cuts = 5
number_of_random_yz_cuts = 5

# minimum amount of labeled voxels within a slice to be considered 'not empty'
roi_is_empty_threshold = 100

####################

all_segmentations = glob.glob(os.path.join(path_to_segmentations, '*.h5'))
shuffle(all_segmentations)

train_ratio_split = 0.7
split_point = int(len(all_segmentations) * train_ratio_split)
train_segmentations = all_segmentations[:split_point]
test_segmentations = all_segmentations[split_point:]

output_folder_test = 'data/extracted_slices_train/'
output_folder_train = 'data/extracted_slices_test/'

def write_3D_segmentations_as_2D_slices(list_of_segmentations, output_folder):
    if not os.path.exists(output_folder):
        os.makedirs(output_folder)

    for path_to_segmentation in tqdm.tqdm(list_of_segmentations):
        stackname = os.path.splitext(os.path.basename(path_to_segmentation))[0]
        path_to_matching_wga = os.path.join(path_to_image_data, f'deblur_{stackname}_c3.h5')

        if not os.path.exists(path_to_matching_wga):
            print(f'couldnt fine wga for {stackname}')
            continue

        segmentation = itk.GetArrayFromImage(itk.imread(path_to_segmentation))
        wga = itk.GetArrayFromImage(itk.imread(path_to_matching_wga))

        #TODO: think about: max downsample for labels good?
        segmentation = block_reduce(segmentation, downsample_factor, np.max)
        wga = downscale_local_mean(wga, downsample_factor).astype(np.uint8)

        if segmentation.shape != wga.shape:
            print(f'shape mismatch:     {segmentation.shape}     {wga.shape}      {stackname}')

        zDim, yDim, xDim = segmentation.shape

        for _ in range(number_of_random_xy_cuts):
            no_segmentation_in_ROI = True
            while no_segmentation_in_ROI:
                random_z = np.random.randint(0, zDim - 1)
                random_cell_seg = segmentation[random_z, :, :]
                no_segmentation_in_ROI = np.sum(segmentation) < roi_is_empty_threshold

            random_wga = wga[random_z, :, :]
            output_path_img = os.path.join(output_folder, f'{stackname}_z_{random_z}.tif')
            output_path_mask = os.path.join(output_folder, f'{stackname}_z_{random_z}_masks.tif')
            itk.imwrite(itk.GetImageFromArray(random_wga), filename=output_path_img)
            itk.imwrite(itk.GetImageFromArray(random_cell_seg.astype(np.uint16)), filename=output_path_mask)

        for _ in range(number_of_random_xz_cuts):
            no_segmentation_in_ROI = True
            while no_segmentation_in_ROI:
                random_y = np.random.randint(0, yDim - 1)
                random_cell_seg = segmentation[:, random_y, :]
                no_segmentation_in_ROI = np.sum(segmentation) < roi_is_empty_threshold

            random_wga = wga[:, random_y, :]
            output_path_img = os.path.join(output_folder, f'{stackname}_y_{random_y}.tif')
            output_path_mask = os.path.join(output_folder, f'{stackname}_y_{random_y}_masks.tif')
            itk.imwrite(itk.GetImageFromArray(random_wga), filename=output_path_img)
            itk.imwrite(itk.GetImageFromArray(random_cell_seg.astype(np.uint16)), filename=output_path_mask)

        for _ in range(number_of_random_yz_cuts):
            no_segmentation_in_ROI = True
            while no_segmentation_in_ROI:
                random_x = np.random.randint(0, xDim - 1)
                random_cell_seg = segmentation[:, :, random_x]
                no_segmentation_in_ROI = np.sum(segmentation) < roi_is_empty_threshold

            random_wga = wga[:, :, random_x]
            output_path_img = os.path.join(output_folder, f'{stackname}_x_{random_x}.tif')
            output_path_mask = os.path.join(output_folder, f'{stackname}_x_{random_x}_masks.tif')
            itk.imwrite(itk.GetImageFromArray(random_wga), filename=output_path_img)
            itk.imwrite(itk.GetImageFromArray(random_cell_seg.astype(np.uint16)), filename=output_path_mask)

write_3D_segmentations_as_2D_slices(train_segmentations, output_folder_train)
write_3D_segmentations_as_2D_slices(test_segmentations, output_folder_test)

with open('data/train_paths.pkl', 'wb') as f:
    pickle.dump(train_segmentations, f)

with open('data/test_paths.pkl', 'wb') as f:
    pickle.dump(test_segmentations, f)