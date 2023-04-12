import os
import sys
import ast

def fileargs2dct(args_file):
    with open(args_file,'r') as file:
        dct = {}
        for line in file.readlines():
            if line[0]=='#':
                continue
            k,v = ast.literal_eval(line)
            dct[k] = v
    return dct

#runs the preprocessing steps above to create nnUnet compatible dat
if __name__ == "__main__":
	print('Run CT lesion segmentation application')
	workingdir = os.path.dirname(sys.argv[0])
	print(workingdir)
	#sys.path.append(workingdir)

	#use arguments that are passed from the argument file
	dct_args = fileargs2dct(sys.argv[1])

	#load variables to run the script
	path_in = dct_args['path_in'] #path of the CT scans
	path_model = dct_args['path_model'] #where is the model root path?
	path_out = dct_args['path_out'] #output directory
	analysis = dct_args['analysis']
	imagename = dct_args['imagename']
	vm_dilate_mm = dct_args['vm_dilate_mm']

	print('Arguments:',dct_args)

	#Run the preprocess command
	cmd_preprocess = 'python {} {} {}'.format(os.path.join(workingdir,'preprocess.py'), 
												path_in, 
												path_out, 
												analysis)
	print(cmd_preprocess)
	os.system(cmd_preprocess)

	# #Run the nnUnet inference script
	cmd_inference = 'python {} {} {} {}'.format(os.path.join(workingdir,'nnunet_inference.py'),
												path_model, #main model directory
												path_out,  #preprocessed form imagesTs should be here in subdir
												analysis #analysis type
												)
	print(cmd_inference)
	os.system(cmd_inference)

	#Run postprocess script
	cmd_postprocess = 'python {} {} {} {}'.format(os.path.join(workingdir,'postprocess.py'), 
													path_out,
													imagename, 
													vm_dilate_mm)
	print(cmd_postprocess)
	os.system(cmd_postprocess)


	