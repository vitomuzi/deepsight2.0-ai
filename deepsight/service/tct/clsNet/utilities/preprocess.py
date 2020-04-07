import numpy as np
import math
import cv2
import SimpleITK as sitk

# data augmentation method
def affine2D(img, x_scale, y_scale, x_shift, y_shift, rad_rotate, rotate_center=None):
	# img is a np array.
	(x_range, y_range) = img.shape()
	m_shift = [
		[1, 0, x_shift], 
		[0, 1, y_shift], 
		[0, 0, 1]]
	m_rotate = [
		[math.cos(rad_rotate), -math.sin(rad_rotate), 0], 
		[math.sin(rad_rotate), math.cos(rad_rotate), 0], 
		[0, 0, 1]]
	m_scale = [
		[x_scale, 0, (1-x_scale)*(x_range/2)],
		[0, y_scale,  (1-y_scale)*(y_range/2)], 
		[0, 0, 1]]

	if rotate_center is None:
		m_rotate_center = np.dot(
			np.array([
				[1,0,x_range/2],
				[0,1,y_range/2],
				[0,0,1]
				]), 
			np.dot(m_rotate, np.array([
				[1,0,-x_range/2],
				[0,1,-y_range/2],
				[0,0,1]])))
	else:
		m_rotate_center = np.dot(
			np.array([
				[1,0,rotate_center[0]],
				[0,1,rotate_center[1]],
				[0,0,1]
				]), 
			np.dot(m_rotate, np.array([
				[1,0,-rotate_center[0]],
				[0,1,-rotate_center[1]],
				[0,0,1]])))


	m_affine = np.dot(m_rotate_center, np.dot(m_shift, m_scale))
	m_affine_cv = m_affine[:2, :]

	img_warpped = cv2.warpAffine(img,m_affine_cv,(x_range, y_range))
	return img_warpped

def affine3D(img, x_scale, y_scale, x_shift, y_shift, rad_rotate):
	pass

def flip(img, axis):
	return np.flip(img, axis)

def add_noise2D():
	pass

def blur():
	pass

# image preprocess
def normalize(img, mean_i=0, std_i=1):
	# exclude some outlier intensity if necessary
	# img[img>np.percentile(img,99.9)] = np.percentile(img,99.9)
	# img[img<np.percentile(img,0.1)]= np.percentile(img,0.1)
	factor_scale = np.std(img)/std_i
	img = img / factor_scale
	factor_shift = np.mean(img) - mean_i
	img = img - factor_shift
	return img

# simple itk image preprocessing
def itk_n4correction(itk_img, mask=None, num_itr=50):
	if mask == None:
		filter_statistic = sitk.StatisticsImageFilter()
		filter_statistic.Execute(itk_img)
		min_intensity = filter_statistic.GetMinimum()
		mask = itk_img > min_intensity

	itk_img = sitk.Cast(itk_img, sitk.sitkFloat32)
	corrector = sitk.N4BiasFieldCorrectionImageFilter()
	corrector.SetMaximumNumberOfIterations(int(num_itr))

	output_itk_img = corrector.Execute(itk_img, mask)
	return output_itk_img

def cpp_n4correction():
	# call compiled cpp program
	pass

def itk_hist_matching(itk_img, itk_template, num_hist_level=1024, num_match_point=7):
	return sitk.HistogramMatching(itk_img, itk_template,
            numberOfHistogramLevels=num_hist_level, numberOfMatchPoints=num_match_point)

# shift, scale and rotate centered with centroid of a 3d image
def itk_similarity_3d(itk_img, output_spacing=None, output_size=None, rotate=None, scale=None, shift=None, interpolate_method='Linear'):
	# interpolate_method = ['NearestNeighbor', 'Linear', 'BSpline']
	# define transform
	if rotate==None:
		rotate = (0,0,0)
	if scale==None:
		scale = (1,1,1)
	if shift==None:
		shift = (0,0,0)
	rotate = tuple(np.array(rotate)*np.pi/180)

	input_size = itk_img.GetSize()
	border_index = tuple(np.array(input_size)-1)
	center_point = tuple((np.array(itk_img.TransformIndexToPhysicalPoint(border_index)) + np.array(itk_img.GetOrigin()))/2)

	rigid_euler = sitk.Euler3DTransform(center_point, rotate[0], rotate[1], rotate[2], shift)
	aniso_scale = sitk.ScaleTransform(3)
	aniso_scale.SetCenter(center_point)
	aniso_scale.SetScale((scale,scale,scale))
	composite_transform = sitk.Transform(aniso_scale)
	composite_transform.AddTransform(rigid_euler)

	# define resampler
	filter_resample = sitk.ResampleImageFilter()
	filter_resample.SetInterpolator(eval('sitk.sitk'+interpolate_method))
	if output_spacing != None:
		filter_resample.SetOutputSpacing(output_spacing)
	else:
		filter_resample.SetOutputSpacing(itk_img.GetSpacing())

	if output_size != None:
		filter_resample.SetSize(output_size)
	else:
		filter_resample.SetSize(itk_img.GetSize())

	filter_resample.SetTransform(composite_transform)
	filter_resample.SetOutputOrigin(itk_img.GetOrigin())
	filter_resample.SetOutputDirection(itk_img.GetDirection())

	output_itk_img = filter_resample.Execute(itk_img)
	return output_itk_img

def get_similarity_transformer_3d(output_spacing=None, output_size=None, rotate=None, scale=None, shift=None, interpolate_method='Linear'):
	# interpolate_method = ['NearestNeighbor', 'Linear', 'BSpline']
	# define transform
	if rotate==None:
		rotate = (0,0,0)
	if scale==None:
		scale = (1,1,1)
	if shift==None:
		shift = (0,0,0)
	rotate = tuple(np.array(rotate)*np.pi/180)

	input_size = itk_img.GetSize()
	border_index = tuple(np.array(input_size)-1)
	center_point = tuple((np.array(itk_img.TransformIndexToPhysicalPoint(border_index)) + np.array(itk_img.GetOrigin()))/2)

	rigid_euler = sitk.Euler3DTransform(center_point, rotate[0], rotate[1], rotate[2], shift)
	aniso_scale = sitk.ScaleTransform(3)
	aniso_scale.SetCenter(center_point)
	aniso_scale.SetScale(scale)
	composite_transform = sitk.Transform(aniso_scale)
	composite_transform.AddTransform(rigid_euler)
	return composite_transform
	