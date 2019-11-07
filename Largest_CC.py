import os
import SimpleITK as sitk
import numpy as np
import skimage
from skimage.measure import label   

# def getLargestCC(segmentation):
#     labels = label(segmentation)
#     largestCC = labels == np.argmax(np.bincount(labels.flat))
#     return largestCC

root_folder = '/media/halm/PROJECT/Projects/liver segmentation/liver_detection'

segmentation_file_name = 'p02_pos_seg_unet.nii'

liver_ROI_post = '-liver'

segmentation_file_path = os.path.join(root_folder, 'pre_processing', segmentation_file_name)


path,name = os.path.split(segmentation_file_path)
name      = name.split('.')
liver_ROI_file_name = name[0] + liver_ROI_post + '.' + name[1]
liver_ROI_file_name_path = os.path.join(root_folder, 'pre_processing',liver_ROI_file_name)

print(segmentation_file_path)

segmentation     = sitk.ReadImage(segmentation_file_path)


# print(type(segmentation))

cc = sitk.ConnectedComponent(segmentation,True)

stats = sitk.LabelIntensityStatisticsImageFilter()
stats.Execute(cc,segmentation)

largestCClabel = 0
largestCCsize = 0 

for l in stats.GetLabels():
    print("Label: {0} -> Mean: {1} Size: {2}".format(l, stats.GetMean(l), int(stats.GetPhysicalSize(l))))
    if int(stats.GetPhysicalSize(l)) >= largestCCsize:
        largestCCsize = int(stats.GetPhysicalSize(l))
        largestCClabel = l
    
largestCC = cc == largestCClabel # get the largest component


# labelImageArray = sitk.GetArrayFromImage(segmentation)

# print(type(labelImageArray))

# labels = label(labelImageArray)

# largestCC = labels == np.argmax(np.bincount(labels.flat))

sitk.WriteImage(largestCC,liver_ROI_file_name_path, False) 

