##the syntax below - based on the download_curl.sh can be used to download individual files - just replace the PAN01 with another subject number#
##this is adjusted for my task glm code, nilearn_epiproj_trials.ipynb##
curl --create-dirs 'https://s3.amazonaws.com/openneuro.org/ds006598/sub-PAN01/ses-4/func/sub-PAN01_ses-4_task-epiproj_space-MNI152NLin6Asym_res-2_desc-preproc_bold.nii.gz?versionId=ruLaVmTA_EBWgYkfUCFXn0HWqLDC7ZMq' -o 'ds006598-download/sub-PAN01/ses-4/func/sub-PAN01_ses-4_task-epiproj_space-MNI152NLin6Asym_res-2_desc-preproc_bold.nii.gz'

##fork this directory with all the 3 column task timing files from my github and then clone it##
ds006598-download/derivatives/afni_timing_epiproj
