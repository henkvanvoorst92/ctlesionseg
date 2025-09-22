
import sys
import os
import numpy as np
import SimpleITK as sitk
from scipy.ndimage.morphology import binary_fill_holes
from tqdm import tqdm
import pandas as pd
from utils import np2itk
from utils import sitk_select_components_minsize
from utils import np_volume
from utils import compute_volume
from utils import np_slicewise
from utils import z_crop_mask
from utils import sitk_erode_mm
from utils import clip_image




def gaussian_filter_mm(image: sitk.Image, sigmas_mm: list, method='recursive'):
    """
    Apply Gaussian smoothing with per-dimension sigma values given in millimeters.

    Parameters
    ----------
    image : sitk.Image
        Input image to smooth.
    sigmas_mm : list of float
        Standard deviations (sigma) for the Gaussian kernel in millimeters, per dimension.
    method : str, optional
        Gaussian filter method; either 'recursive' (default) or 'discrete'.

    Returns
    -------
    sitk.Image
        The Gaussian-filtered image.
    """
    if method == 'recursive':
        # Fast IIR-based Gaussian smoothing
        return sitk.SmoothingRecursiveGaussian(image, sigma=sigmas_mm)
    elif method == 'discrete':
        # Convolution-based Gaussian smoothing
        # Note: DiscreteGaussian expects variance per dimension
        variances = [s ** 2 for s in sigmas_mm]
        return sitk.DiscreteGaussian(image, variance=variances)
    else:
        raise ValueError("method must be either 'recursive' or 'discrete'")

def Resample_img(img, new_spacing=0.45, interpolator=sitk.sitkLinear):
    #sitkBSplineResamplerOrder3
    # new_spacing should be in sitk order x,y,z (np order: z,y,x)
    if isinstance(new_spacing,int) or isinstance(new_spacing,float):
       new_spacing = [new_spacing,new_spacing,new_spacing]
    #https://github.com/SimpleITK/SimpleITK/issues/561
    resample = sitk.ResampleImageFilter()
    resample.SetInterpolator = interpolator
    resample.SetOutputDirection(img.GetDirection()) 
    resample.SetOutputOrigin(img.GetOrigin())
    resample.SetOutputSpacing(new_spacing)
    new_size = new_img_size(img,new_spacing)
    resample.SetSize(new_size)
    img = resample.Execute(img)
    img = sitk.Cast(img, sitk.sitkInt32)
    return img

def Resample_slices(img, new_z_spacing=5, interpolator=sitk.sitkLinear):
    #https://github.com/SimpleITK/SimpleITK/issues/561
    resample = sitk.ResampleImageFilter()
    resample.SetInterpolator = interpolator
    resample.SetOutputDirection(img.GetDirection()) 
    resample.SetOutputOrigin(img.GetOrigin())
    new_spacing = [*img.GetSpacing()[:2],new_z_spacing]
    resample.SetOutputSpacing(new_spacing)
    new_n_slices = int(np.ceil(img.GetSize()[-1]*img.GetSpacing()[-1]/new_z_spacing))
    resample.SetSize((*img.GetSize()[:2], new_n_slices))
    img = resample.Execute(img)
    img = sitk.Cast(img, sitk.sitkInt32)
    return img




#runs the preprocessing steps above to create nnUnet compatible dat
if __name__ == "__main__":
	print('Preprocessing all NCCTs to be compatible with nnUnet input')
	path_in = sys.argv[1] #import as argument
	path_out = sys.argv[2] #import as argument
	analysis = sys.argv[3]

	#set other variables
	if len(sys.argv)>4:
		add_ventriclemask = sys.argv[4] #true or false --> will include use of ventricle mask
	elif len(sys.argv)>5:
		vm_dilate = sys.argv[5] #int --> if>0 will include dilate ventricle mask for lesion selection
	elif len(sys.argv)>6:
		top_mm_rm, bottom_mm_rm = sys.argv[6], sys.argv[6]
	elif len(sys.argv)>7:
		erode_mm_xy = sys.argv[7]
	elif len(sys.argv)>8:
		clip_range = sys.argv[8]
	else:
		if analysis=='wml':
			#Default variables
			add_ventriclemask = True
			vm_dilate_mm = 10 #dilation in mm of the ventricle mask (vm)
			min_ventricle_vol = 1500 #in mm3
			modality = '0000'
			top_mm_rm, bottom_mm_rm = 40, 40 #removes mm's from top and bottom of brainmask slices
			erode_mm_xy = 20 #erode brain mask in xy dims
			clip_range = (0,100)
		else: #for infarct lesion segmentation
			add_ventriclemask = False
			vm_dilate_mm = 0 #dilation in mm of the ventricle mask (vm)
			min_ventricle_vol = None #in mm3
			modality = '0000'
			top_mm_rm, bottom_mm_rm = 0, 0 #removes mm's from top and bottom of brainmask slices
			erode_mm_xy = 0 #erode brain mask in xy dims
			clip_range = (0,1000)

	sitk_type = sitk.sitkInt16

	p_img_ts = os.path.join(path_out,'imagesTs')
	p_bm_ts = os.path.join(path_out,'brainmask')

	if not os.path.exists(p_img_ts):
		os.makedirs(p_img_ts)
	if not os.path.exists(p_bm_ts):
		os.makedirs(p_bm_ts)

	if add_ventriclemask:
		p_vm_ts = os.path.join(path_out,'ventriclemask')
		#p_vm_dil_ts = os.path.join(path_out,'ventriclemaskdilTs')
		if not os.path.exists(p_vm_ts):
			os.makedirs(p_vm_ts)
		#if not os.path.exists(p_vm_dil_ts):
		#    os.makedirs(p_vm_dil_ts)

	#iterate over all NCCTs in the root dir
	p_img_ts = os.path.join(path_out,'imagesTs')
	p_bm_ts = os.path.join(path_out,'brainmask')

	if not os.path.exists(p_img_ts):
		os.makedirs(p_img_ts)
	if not os.path.exists(p_bm_ts):
		os.makedirs(p_bm_ts)

	if add_ventriclemask:
		p_vm_ts = os.path.join(path_out,'ventriclemask')
		#p_vm_dil_ts = os.path.join(path_out,'ventriclemaskdilTs')
		if not os.path.exists(p_vm_ts):
			os.makedirs(p_vm_ts)
		#if not os.path.exists(p_vm_dil_ts):
		#    os.makedirs(p_vm_dil_ts)

	#iterate over all NCCTs in the root dir
	out = []
	for f_ncct in tqdm(os.listdir(path_in)):
		if not '.nii' in f_ncct:
			continue
		ID = f_ncct.split('.')[0]
		p_ncct = os.path.join(path_in,f_ncct)
		NCCT= sitk.Cast(sitk.ReadImage(p_ncct), sitk_type)
		print(ID,'NCCT imported', NCCT.GetSize())

		z_spacing = NCCT.GetSpacing()[-1]
		print(f'{ID}, z-spacing: {z_spacing}')
		if z_spacing < 3.0:
			print(f'{ID}, z-spacing < 3.0 ({z_spacing})')
			# downsample thin slices to 4.0 mm
			new_spacing = 4.0
			for i in range(2,10):
					# try to fit a nice multiple of the current spacing
					if i*z_spacing >= 3.5 and i*z_spacing < 5.0:
						new_spacing = i*z_spacing
			# Only blur along z
			print(f'{ID}, blurring with sigma of {new_spacing/2.0}')
			NCCT_blurred = gaussian_filter_mm(NCCT, [0.0, 0.0, new_spacing/2.0])
			print(f'{ID}, resampling to slice spacing of {new_spacing}')
			NCCT_resampled = Resample_slices(NCCT_blurred, new_spacing)
			NCCT = NCCT_resampled

		#create a (simple) brain mask (BM)
		BM = sitk.BinaryThreshold(NCCT, lowerThreshold=0, upperThreshold=150,
										  insideValue=1, outsideValue=0)
		BM = np_slicewise(sitk.GetArrayFromImage(BM), [binary_fill_holes])
		BM = z_crop_mask(np2itk(BM,NCCT),top_mm_rm,bottom_mm_rm)
		BM_er = sitk.GetArrayFromImage(sitk_erode_mm(BM,kernel_mm=[erode_mm_xy,erode_mm_xy,0]))
		BM = sitk.Cast(BM, sitk_type)

		print(ID,'BM done')
		if add_ventriclemask:
			CSF_mask = sitk.BinaryThreshold(NCCT, lowerThreshold=-10, upperThreshold=17,
												  insideValue=1, outsideValue=0)
			#get the largest connected components (LCC) from the CSF
			LCC_mask = sitk_select_components_minsize(CSF_mask,min_vol_ml=1)
			LCC_mask = sitk.GetArrayFromImage(LCC_mask)*BM_er

			ventricle_mask = np_slicewise(LCC_mask, [binary_fill_holes])
			ventricle_volume = np_volume(ventricle_mask,NCCT.GetSpacing())

			#simpler method if ventricle volume is to low
			if ventricle_volume<min_ventricle_vol:
				CSF_mask = sitk.BinaryThreshold(NCCT, lowerThreshold=-20, upperThreshold=20,
											  insideValue=1, outsideValue=0)
				LCC_mask = sitk.Cast(sitk_select_components_minsize(CSF_mask,min_vol_ml=1), sitk_type)
				LCC_mask = LCC_mask*BM
				LCC_mask = sitk.GetArrayFromImage(LCC_mask)
				ventricle_mask = np_slicewise(LCC_mask, [binary_fill_holes])

			VM = sitk.Cast(np2itk(ventricle_mask, NCCT), sitk_type)
		print(ID, 'VM done')
		NCCT = sitk.Cast(clip_image(NCCT,*clip_range), sitk_type)

		img_name = ID +'_'+modality+'.nii.gz'
		label_name = ID +'.nii.gz'

		p_ncct = os.path.join(p_img_ts,img_name)
		sitk.WriteImage(NCCT,p_ncct)
		p_bm = os.path.join(p_bm_ts,label_name)
		sitk.WriteImage(BM,p_bm)

		if add_ventriclemask:
			p_vm = os.path.join(p_vm_ts,label_name)
			sitk.WriteImage(VM,p_vm)
			#p_vm_dil = os.path.join(p_vm_dil_ts,label_name)
			#sitk.WriteImage(VM_dil,p_vm_dil)
		else:
			p_vm = None

		out.append([ID,p_ncct,p_bm,p_vm, img_name, label_name])

		df = pd.DataFrame(out)

		df.columns = ['ID','p_ncct','p_bm','p_vm','img_name','label_name']
		df.index = df.ID
		df.to_excel(os.path.join(path_out,'preproccesed_images_report.xlsx'))

