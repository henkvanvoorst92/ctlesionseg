import os
import sys
#import nnunet

#runs the preprocessing steps above to create nnUnet compatible dat
if __name__ == "__main__":
	print('Running nnUnet for Inference')
	#use arguments that are passed
	root_model = sys.argv[1] #where is the model root path? r'/data/projects/stroke/nnUnet_vol_vessels' #where model is
	root_data = sys.argv[2] #import as argument (where images are)
	path_in = os.path.join(root_data,'ImagesTs')
	path_out = os.path.join(root_data,'predicted')
	if not os.path.exists(path_out):
	    os.makedirs(path_out)
	#root folder with data for nnunet

	#required model settins
	analysis = sys.argv[3]
	if analysis=='wml':
		datano = '525'
		task = 'Task525_WMLCT'
		res = '2d'
	elif analysis=='24h_infarct':
		datano = '503'
		task = 'Task503_infarct24h'
		res = '2d'
	elif analysis=='1w_infarct':
		datano = '502'
		task = 'Task502_infarct1week'
		res = '2d'
	else:
		print('Error wrong analysis type',analysis)

	# datano = sys.argv[3] 
	# task = sys.argv[4]
	# res = sys.argv[5]

	#fix path for model 
	os.environ['nnUNet_raw_data_base']= os.path.join(root_model,'nnUNet_raw_data_base')
	os.environ['nnUNet_preprocessed'] = os.path.join(root_model,'nnUNet_preprocessed')
	os.environ['RESULTS_FOLDER'] = os.path.join(root_model,'nnUNet_trained_models')
	os.environ['MKL_THREADING_LAYER'] = 'GNU'

	print('Finding best configuration')
	cmd = 'nnUNet_find_best_configuration -m {}-t {}'.format(res,datano) # --strict
	os.system(cmd)

	print('Starting inference')
	cmd = 'nnUNet_predict -i {} -o {} -tr nnUNetTrainerV2 -ctr \
	    nnUNetTrainerV2CascadeFullRes -m {} -p \
	    nnUNetPlansv2.1 -t {}'.format(path_in,path_out,res,task)#--save_npz

	print(cmd)
	os.system(cmd)



	
