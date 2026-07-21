#!/bin/env python3

import os, re
import bids
from nilearn import image as nimg
from nilearn import plotting as nplot
import nilearn.glm.first_level
import pandas as pd
import numpy as np
import logging
import argparse
import glob

logging.basicConfig(level=logging.INFO,
                    format='%(asctime)s - %(levelname)s - %(message)s',
                    datefmt='%Y-%m-%d %H:%M:%S')

logger = logging.getLogger("nilearn.task")

parser = argparse.ArgumentParser(description='run a GLM analysis with nilearn')
parser.add_argument("--subj", dest="subjid", help="BIAC subject number", required=True)
parser.add_argument("--sess", dest="session", help="mri scanning session (1, 2, etc)", required=True)

args = parser.parse_args()

subj = int(args.subjid)
sess = int(args.session)

#some experiment info
expname = "NEURONIC.01"
muninshare = "munin2"
task='gng'
basedir = os.path.join("/mnt",muninshare,"Sweitzer",expname)

teddir = os.path.join(basedir,'Analysis/preprocessing/tedana')
fpdir = os.path.join(basedir,'Analysis/preprocessing/fmriprep/FAST')
stimdir = os.path.join(basedir,'Stimuli/SubjectData/GNG')
outdir = os.path.join(basedir,'Analysis/task/niilearn/gng/withretroicor/tedana/sub-%04i' % subj)
os.makedirs(outdir, exist_ok=True)

func_files = glob.glob(os.path.join(teddir,'sub-%04i' %subj,'ses-01/func/*MNI*optcomDenoised*.nii.gz'))
func_files = [f for f in func_files if task in f]
mask_files = glob.glob(os.path.join(fpdir,'sub-%04i' %subj,'ses-01/func/*MNI*mask*.nii.gz'))
mask_files = [f for f in mask_files if task in f]
confound_files = glob.glob(os.path.join(fpdir,'sub-%04i' %subj,'ses-01/func/*confounds*.tsv'))
confound_files = [f for f in confound_files if task in f]
events_files = glob.glob(os.path.join(stimdir,'%04i' %subj,'*.tsv'))
events_files = [f for f in events_files if 'use.tsv' in f]

#append data for fixed effects model
design_matrices = []
fmri_img = []
masks = []

func_tr = 2.0
confound_vars = ['trans_x','trans_y','trans_z',
                 'rot_x','rot_y','rot_z']
confound_vars += ["cosine%02i" % (i,) for i in range(3)]

#grab the the necessary files to run the models
for idx,(funcf,maskf,conff,onsf) in enumerate(zip(func_files,mask_files,confound_files,events_files)):
    
    #get the confounds
    confound_data = pd.read_table(conff)
    nonsteady = [col for col in confound_data if col.startswith('non_steady_state_outlier')]
    these_confs = confound_data[confound_vars].fillna(0)
    these_confs = these_confs.iloc[len(nonsteady):] ##subtract volumes##
    these_confs.to_csv(os.path.join(outdir,"run%02i_confs.csv" % (idx+1,)))
    
    #get the func
    func_img = nimg.load_img(funcf)
    fmri_img.append(func_img)

    #calc mean for display
    mask_img = nimg.load_img(maskf)
    masks.append(mask_img)

    #read fmri volumes in seconds
    frame_times = np.arange(func_img.shape[-1]) * func_tr

    #read the onsets and truncate per non-steady volumes
    ons = pd.read_table(onsf)
    ons.onset -= (func_tr * len(nonsteady)) ##subtract volumes##
    ons = ons[ons.onset > 0]

    #group trials per condition - makes a 3 column matrix for each condition##
    freqcor = ons[(ons.trial_type == "freqgo_cor")].index.values  
    freqcor_3col = ons.loc[freqcor, ["onset", "duration","trial_type"]]

    freqinc = ons[(ons.trial_type == "freqgo_inc")].index.values
    freqinc_3col = ons.loc[freqinc, ["onset", "duration","trial_type"]]
    if len(freqinc_3col) == 0:
       freqinc_3col = pd.DataFrame(0, index=np.arange(1), columns=freqinc_3col.columns)
       freqinc_3col['trial_type'][0]='freqgo_inc'   

    infreqcor = ons[(ons.trial_type == "infreqgo_cor")].index.values
    infreqcor_3col = ons.loc[infreqcor, ["onset", "duration","trial_type"]]
    if len(infreqcor_3col) == 0:
       infreqcor_3col = pd.DataFrame(0, index=np.arange(1), columns=infreqcor_3col.columns)
       infreqcor_3col['trial_type'][0]='infreqgo_cor' 
    
    infreqinc = ons[(ons.trial_type == "infreqgo_inc")].index.values
    infreqinc_3col = ons.loc[infreqinc, ["onset", "duration","trial_type"]]
    if len(infreqinc_3col) == 0:
       infreqinc_3col = pd.DataFrame(0, index=np.arange(1), columns=infreqinc_3col.columns)
       infreqinc_3col['trial_type'][0]='infreqgo_inc' 
    
    nogocor = ons[(ons.trial_type == "nogo_cor") & (ons.newacc == 1)].index.values
    nogocor_3col = ons.loc[nogocor, ["onset", "duration","trial_type"]]
    if len(nogocor_3col) == 0:
       nogocor_3col = pd.DataFrame(0, index=np.arange(1), columns=nogocor_3col.columns)
       nogocor_3col['trial_type'][0]='nogo_cor' 
    
    nogoinc = ons[(ons.trial_type == "nogo_inc") & (ons.newacc == 0)].index.values
    nogoinc_3col = ons.loc[nogoinc, ["onset", "duration","trial_type"]]  
    if len(nogoinc_3col) == 0:
       nogoinc_3col = pd.DataFrame(0, index=np.arange(1), columns=nogoinc_3col.columns)
       nogoinc_3col['trial_type'][0]='nogo_inc' 
    
    nogoom = ons[(ons.trial_type == "nogo_om")].index.values
    nogoom_3col = ons.loc[nogoom, ["onset", "duration","trial_type"]]
    if len(nogoom_3col) == 0:
       nogoom_3col = pd.DataFrame(0, index=np.arange(1), columns=nogoom_3col.columns)
       nogoom_3col['trial_type'][0]="nogo_om" 
    
    ##combine the matrices into a single list of events##
    events = pd.concat([nogocor_3col,infreqcor_3col,freqcor_3col,nogoinc_3col,infreqinc_3col,freqinc_3col,nogoom_3col])
    events.to_csv(os.path.join(outdir,"run%02i_events.csv" % (idx+1,)))

    #make the design matrix from all the above file - one per run#
    design = nilearn.glm.first_level.make_first_level_design_matrix(frame_times, events,drift_model=None,high_pass=0.01, add_regs=these_confs, hrf_model='spm + derivative')
    design_matrices.append(design)
    design.to_csv(os.path.join(outdir,"run%02i_design.csv" % (idx+1,)))


#mean the masks
mean_mask = nimg.binarize_img(nimg.mean_img(masks))
#mean the func image for background
bg_img = nimg.mean_img(fmri_img)

fmri_glm = nilearn.glm.first_level.FirstLevelModel(mask_img=mean_mask,n_jobs=8,smoothing_fwhm=4,
                t_r=int(func_tr))

#setup basic contrasts
#fill 0s in design_matrices to even columns#
filler = [col for col in design_matrices[0].columns if col not in design_matrices[1].columns]
design_matrices[1][filler]=0.0
filler = [col for col in design_matrices[1].columns if col not in design_matrices[0].columns]
design_matrices[0][filler]=0.0
print(design_matrices)
contrast_matrix = np.eye(design.shape[1])
basic_contrasts = dict([(column, contrast_matrix[i]) for i, column in enumerate(design.columns)])

#add more contrasts
contrasts = { 'nogocor_minus_infreqgocor' : basic_contrasts['nogo_cor'] - basic_contrasts['infreqgo_cor'],  
              'infreqgocor_minus_nogocor' : basic_contrasts['infreqgo_cor'] - basic_contrasts['nogo_cor'],
              'nogocor_minus_nogoinc' : basic_contrasts['nogo_cor'] - basic_contrasts['nogo_inc'],
              'nogoinc_minus_nogocor' : basic_contrasts['nogo_inc'] - basic_contrasts['nogo_cor'],
              'infreqgocor_minus_freqgocor' : basic_contrasts['infreqgo_cor'] - basic_contrasts['freqgo_cor'],
}
 
#run the glm and save report
fmri_glm = fmri_glm.fit(fmri_img,design_matrices=design_matrices)
report = fmri_glm.generate_report(contrasts,threshold=2.0)
report.save_as_html(os.path.join(outdir,'%s_report.html' % task))

#do a fixed effects model combining the runs
for index, (contrast_id, contrast_val) in enumerate(contrasts.items()):
    print('  Contrast % 2i out of %i: %s' % (index + 1, len(contrasts), contrast_id))
    # Estimate the contasts. Note that the model implicitly computes a fixed
    # effect across the two sessions
    z_map = fmri_glm.compute_contrast(contrast_val, output_type='z_score')
    # write the resulting stat images to file
    z_image_path = os.path.join(outdir, '%s_%s_z_map.nii.gz' % (task,contrast_id))
    z_map.to_filename(z_image_path)
    display=nplot.plot_stat_map(z_map, bg_img=bg_img, threshold=2.0, title="%s" % (contrast_id,), display_mode='z', cut_coords=3, black_bg=True)
    display.savefig(os.path.join(outdir,"%s_%s.png" % (task,contrast_id,)))

##show individual run effects?
#for idx in range(len(fmri_img)):
#     outdirrun = os.path.join(outdir, 'run-%02i' % (idx + 1))
#     os.makedirs(outdirrun, exist_ok=True)
#     glm = fmri_glm.fit(fmri_img[idx], design_matrices=design_matrices[idx])
#     for con in contrasts:
#         z_map = glm.compute_contrast(contrasts[con], output_type='z_score')
#         z_image_path = os.path.join(outdirrun, '%s_run-%02i_z_map.nii.gz' % (con, idx + 1))
#         z_map.to_filename(z_image_path)
#         display=nplot.plot_stat_map(z_map,threshold=2.0, title="%s run-%02i" % (con, idx + 1))
#         display.savefig(os.path.join(outdirrun,"%s_run%02i.png" % (con, idx + 1)))


