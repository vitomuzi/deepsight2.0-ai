from utilities import preprocess as prp
import matplotlib.pyplot as plt
import random
import argparse

import numpy as np
import SimpleITK as sitk

# if __name__ == '__main__':
# 	parser = argparse.ArgumentParser(description='Demo of toolkit.');
# 	parser.add_argument('-d', '--dir', type=str, default='/home/lc/data/trabData/No6HospitalData/denosing');
# 	parser.add_argument('--suffix', type=str, default='.nii.gz');
# 	args = parser.parse_args();

# 	data_path = get_data_path(args.dir, args.suffix)
# 	data_path_dict = data_set_split(data_path, val_percent=0.1, seed_random=1)

# 	data_list = load_img(args.dir, data_path_dict['train'], '.nii.gz')

# 	for i, batch_data in enumerate(batch(data_list, 5)):
# 		# data_list = load_img(args.dir, batch_data)
# 		print('%d/%d' % (i*5, len(data_list)))

file_path = 'D:/data/csi/01/01_wat.nii'
itk_img_gt = sitk.ReadImage(file_path)
img = sitk.GetArrayFromImage(itk_img_gt)


rotate_max = 5
(z_rotate,x_rotate,y_rotate) = (random.uniform(-rotate_max, rotate_max),random.uniform(-rotate_max, rotate_max),random.uniform(-rotate_max, rotate_max))
shift_max = 10
shift_max_z = 6
(z_shift,x_shift,y_shift) = (random.uniform(-shift_max_z, shift_max_z),random.uniform(-shift_max, shift_max),random.uniform(-shift_max, shift_max))
scale_max = 1.1
scale_min = 0.9
scale = random.uniform(scale_min, scale_max)
scale = (scale, scale, scale)
transformer = prp.get_similarity_transformer_3d(itk_img_gt, rotate=(z_rotate,x_rotate,y_rotate), scale=scale, shift=(z_shift,x_shift,y_shift))

# get coordinate info
img_size = itk_img_gt.GetSize()
img_spacing = itk_img_gt.GetSpacing()
resample_size = [36, 128, 128]
resample_scale = tuple(np.array(resample_size).astype(np.float)/np.array(img_size))
resample_spacing = tuple(np.array(img_spacing).astype(np.float)/np.array(resample_scale))

interpolate_method='Linear'
resampler = sitk.ResampleImageFilter()
resampler.SetReferenceImage(itk_img_gt)
resampler.SetSize(resample_size)
resampler.SetOutputSpacing(resample_spacing)
resampler.SetInterpolator(eval('sitk.sitk'+interpolate_method))
resampler.SetTransform(transformer) 
# transformer = prp.get_similarity_transformer_3d(itk_img, rotate=None, scale=(scale,scale,scale), shift=(z_shift,x_shift,y_shift))
# test = prp.itk_similarity_3d(itk_img, output_spacing=[1.25, 1.25, 1.25], output_size=[58, 256, 256], rotate=(0, 0, 30), scale=1, shift=(0, 0, 0), interpolate_method='Linear')
itk_img_gt = resampler.Execute(itk_img_gt)

sitk.WriteImage(itk_img_gt, 'D:/test1.nii')
img_resample = sitk.GetArrayViewFromImage(itk_img_gt)
plt.imshow(img_resample[:,:,12])

itk_img_gt = sitk.ReadImage('D:/data/csi/01/01_wat.nii')
img_from_array = sitk.GetImageFromArray(img)
img_from_array.CopyInformation(itk_img_gt)
sitk.WriteImage(img_from_array, 'D:/test2.nii')