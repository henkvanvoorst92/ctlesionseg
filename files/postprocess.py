import os
import sys
import numpy as np
import SimpleITK as sitk
from tqdm import tqdm
import pandas as pd
import numpy as np
from skimage.segmentation import watershed
import ast
from utils import np2itk
from utils import compute_volume
from utils import sitk_dilate_mask


#runs the preprocessing steps above to create nnUnet compatible dat
if __name__ == "__main__":
	print('Postprocessing all results')

	path = sys.argv[1] #path where all nnUnet predictions are stored
	p_pred = os.path.join(path,'predicted')

	p_res = os.path.join(path,'results')
	if not os.path.exists(p_res):
		os.makedirs(p_res)
	
	#Default variables
	modality = '0000'
	imagename = sys.argv[2]
	vm_dilate_mm = ast.literal_eval(sys.argv[3])
	if vm_dilate_mm >0:
		use_ventriclemask = True
	else:
		use_ventriclemask = False
	
	p_img_ts = os.path.join(path,'imagesTs')
	p_bm_ts = os.path.join(path,'brainmask')
	paths_in = [p_pred,p_img_ts,p_bm_ts]

	if use_ventriclemask:
		p_vm_ts = os.path.join(path,'ventriclemask')
		p_vm_dil_ts = os.path.join(path,'ventriclemaskdilTs')
		paths_in.extend([p_vm_ts,p_vm_dil_ts])

	sitk_type = sitk.sitkInt16

	out = []
	for f_ncct in tqdm(os.listdir(p_img_ts)):
		print(f_ncct)
		if not '.nii' in f_ncct:
			continue
		ID = f_ncct.split('.')[0].replace('_'+modality, '')

		img_name = ID +'_'+modality+'.nii.gz'
		label_name = ID +'.nii.gz'

		p_pred_ID = os.path.join(p_pred,label_name)
		if not os.path.exists(p_pred_ID):
			continue

		pid_out = os.path.join(p_res,ID)
		if not os.path.exists(pid_out):
			os.makedirs(pid_out)
			
		PRED = sitk.Cast(sitk.ReadImage(p_pred_ID), sitk_type)
		p_pred_out = os.path.join(pid_out,ID+'_pred.nii.gz')
		sitk.WriteImage(PRED,p_pred_out)
		
		p_ncct_in = os.path.join(p_img_ts,img_name)
		NCCT= sitk.Cast(sitk.ReadImage(p_ncct_in), sitk_type)
		p_ncct_out = os.path.join(pid_out,ID+'{}.nii.gz'.format('_'+imagename))
		sitk.WriteImage(NCCT,p_ncct_out)

		p_bm_in = os.path.join(p_bm_ts,label_name)
		BM = sitk.Cast(sitk.ReadImage(p_bm_in), sitk_type)
		p_bm_out = os.path.join(pid_out,ID+'_brain_mask.nii.gz')
		sitk.WriteImage(BM,p_bm_out)
		
		#compute volumes for output
		totvol = compute_volume(PRED)
		if use_ventriclemask:
			p_vm_in = os.path.join(p_vm_ts,label_name)
			#p_vm_dil = os.path.join(p_vm_dil_ts,label_name)
			VM = sitk.Cast(sitk.ReadImage(p_vm_in), sitk_type)
			p_vm_out = os.path.join(pid_out,ID+'_ventricle_mask.nii.gz')
			sitk.WriteImage(VM,p_vm_out)

			VM_dil = sitk_dilate_mask(VM,vm_dilate_mm, dilate_2D=True)*BM
			p_vm_dil_out = os.path.join(pid_out,ID+'_ventricle_mask_dil_{}mm.nii.gz'.format(vm_dilate_mm))
			sitk.WriteImage(VM_dil,p_vm_dil_out)

			PRED_VM_DIL = VM_dil*PRED
			p_pred_vm_dil_out = os.path.join(pid_out,ID+'_pred_vm_dil_{}mm.nii.gz'.format(vm_dilate_mm))
			sitk.WriteImage(PRED_VM_DIL,p_pred_vm_dil_out)
			lesion_in_vm_vol = compute_volume(PRED_VM_DIL)
			
			#do watershed of seg to get larger lesion region inside dil
			pred = sitk.GetArrayFromImage(PRED)
			pred_vm_dil = sitk.GetArrayFromImage(PRED_VM_DIL)
			pred_wtr = watershed(pred,pred_vm_dil,mask=pred)
			PRED_WTR = sitk.Cast(np2itk(pred_wtr,PRED),sitk_type)
			p_pred_watershed = os.path.join(pid_out,ID+'_pred_watershed.nii.gz')
			sitk.WriteImage(PRED_WTR, p_pred_watershed)
			lesion_watershed_vol = compute_volume(PRED_WTR)
		else:
			lesion_in_vm_vol,lesion_watershed_vol = np.NaN, np.NaN
			
		row = [ID, f_ncct, p_pred_ID, pid_out, totvol, lesion_in_vm_vol,lesion_watershed_vol]
		print(row)
		out.append(row)
		
	df = pd.DataFrame(out, columns=['ID','f_ncct','p_pred_nnUnet','results_folder',
									'total_lesion_volume', 'lesion_in_vm',
									'lesion_in_vm_after_watershed'])
	df.to_excel(os.path.join(p_res,'volumes.xlsx'))
