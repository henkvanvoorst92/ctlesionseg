
import sys
import os
import numpy as np
import SimpleITK as sitk
from scipy.ndimage.morphology import binary_fill_holes 
#	from scipy.ndimage import watershed_ift

def np2itk(arr,original_img):
	img = sitk.GetImageFromArray(arr)
	img.SetSpacing(original_img.GetSpacing())
	img.SetOrigin(original_img.GetOrigin())
	img.SetDirection(original_img.GetDirection())
	# this does not allow cropping (such as removing thorax, neck)
	#img.CopyInformation(original_img) 
	return img

def sitk_select_components_minsize(mask, #sitk mask to extract connected components from
									min_vol_ml=1):
	component_image = sitk.ConnectedComponent(mask)
	SC= sitk.RelabelComponent(component_image, sortByObjectSize=True)
	sc = sitk.GetArrayFromImage(SC)
	ccs2use = []
	for cc in np.unique(sc)[1:]:
		vol = compute_volume(SC==cc)
		ccs2use.append(cc)
		if vol<min_vol_ml:
			break
	lcc = (SC>0) & (SC <= ccs2use[-1])
	return lcc

def np_volume(mask,spacing):
	vol_per_vox = np.product(np.array(spacing))
	return vol_per_vox*mask.sum()

def compute_volume(mask):
	# mask is an sitk image
	# used to compute the volume in ml for foreground
	sp = mask.GetSpacing()
	vol_per_vox = sp[0]*sp[1]*sp[2]
	
	m = sitk.GetArrayFromImage(mask)
	voxels = m.sum()
	#volume in ml
	tot_volume = vol_per_vox*voxels/1000
	return tot_volume

def np_slicewise(mask, funcs, repeats=1, dim=0):
	"""
	Applies a list of functions iteratively (repeats) slice by slice of an 3D np volume
	mask: mask to do operation on
	funcs: list of functions applied consecutively
	repeats: each function is applied for a set number
	dim: dimension to do operation over (default is z dim)
	"""
	if isinstance(mask,sitk.SimpleITK.Image):
		mask = sitk.GetArrayFromImage(mask)

	out = np.zeros_like(mask)
	for sliceno in range(mask.shape[dim]):
		if dim==0:
			m = mask[sliceno,:,:]
		elif dim==1:
			m = mask[:,sliceno,:]
		elif dim==2:
			m = mask[:,:,sliceno]
		for r in range(repeats):
			for func in funcs:
				m = func(m)
		if dim==0:
			out[sliceno,:,:] = m
		elif dim==1:
			out[:,sliceno,:] = m
		elif dim==2:
			out[:,:,sliceno] = m
	return out

def z_crop_mask(MASK,top_mm_rm,bottom_mm_rm):
	#crops slices in z_dim
	mask = sitk.GetArrayFromImage(MASK)
	__,__, zsp = MASK.GetSpacing()
	top_crop_slices = int(top_mm_rm/zsp)
	bottom_crop_slices = int(bottom_mm_rm/zsp)
	#get min and max mask slice
	mask_slices = np.where(mask.argmax(axis=1).argmax(axis=1)>0)[0]
	if len(mask_slices)>0:
		crop_min = min(mask_slices)+bottom_crop_slices
		crop_max = max(mask_slices)-top_crop_slices
		MASK[:,:,:crop_min] = 0
		MASK[:,:,crop_max:] = 0
	return MASK

def sitk_erode_mm(mask,kernel_mm, background=0, foreground=1):
	
	if isinstance(kernel_mm,int):
		k0 = k1 = k2 = kernel_mm
	elif isinstance(kernel_mm,tuple) or isinstance(kernel_mm,list):
		k0, k1, k2 = kernel_mm
	
	kernel_rad = (int(np.floor(k0/mask.GetSpacing()[0])),
			 int(np.floor(k1/mask.GetSpacing()[1])),
			 int(np.floor(k2/mask.GetSpacing()[2])))
	
	erode = sitk.BinaryErodeImageFilter()
	erode.SetBackgroundValue(background)
	erode.SetForegroundValue(foreground)
	erode.SetKernelRadius(kernel_rad)
	return erode.Execute(mask)

def clip_image(img, min = -1024, max = 1900):
	'''
	clip img between HU values
	:param img: sitk image to be clipped
	:param min: minimum clip value
	:param max: maximum clip value
	:return:
	'''
	clampFilter = sitk.ClampImageFilter()
	if max!=None:
		clampFilter.SetUpperBound(max)
	if min!=None:
		clampFilter.SetLowerBound(min)
	return clampFilter.Execute(img)	

def sitk_dilate_mask(mask,radius_mm, dilate_2D=False):

    radius_3d = [int(np.floor(radius_mm / mask.GetSpacing()[0])),
             int(np.floor(radius_mm / mask.GetSpacing()[1])),
             int(np.floor(radius_mm / mask.GetSpacing()[2]))]
    if dilate_2D:
        radius_3d[2] = 0

    dilate = sitk.BinaryDilateImageFilter()
    dilate.SetBackgroundValue(0)
    dilate.SetForegroundValue(1)
    dilate.SetKernelRadius(radius_3d)
    return dilate.Execute(mask)