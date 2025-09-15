import os

import numpy as np
import dcmri as dc
import napari
import matplotlib.pyplot as plt
import dbdicom as db
import miblab
import mdreg
import vreg


PATH = "C:\\Users\\md1spsx\\Documents\\Data\\SheffieldLungs"


datapath = os.path.join(PATH, 'TestData')
resultspath = os.path.join(PATH, 'TestDataAnalysis')
imagepath = os.path.join(PATH, 'TestDataImages')
checkpointspath = os.path.join(PATH, 'TestDataCheckpoints')
os.makedirs(imagepath, exist_ok=True)
os.makedirs(checkpointspath, exist_ok=True)

SUBJ = 'SheffieldSubject'
DATA = [resultspath, SUBJ, 'Data']
MAPS = [resultspath, SUBJ, 'Maps']
MASKS = [resultspath, SUBJ, 'Masks']

TOTAL_MR = {
    1: "spleen",
    2: "kidney_right",
    3: "kidney_left",
    4: "gallbladder",
    5: "liver",
    6: "stomach",
    7: "pancreas",
    8: "adrenal_gland_right",
    9: "adrenal_gland_left",
    10: "lung_left",
    11: "lung_right",
    12: "esophagus",
    13: "small_bowel",
    14: "duodenum",
    15: "colon",
    16: "urinary_bladder",
    17: "prostate",
    18: "sacrum",
    19: "vertebrae",
    20: "intervertebral_discs",
    21: "spinal_cord",
    22: "heart",
    23: "aorta",
    24: "inferior_vena_cava",
    25: "portal_vein_and_splenic_vein",
    26: "iliac_artery_left",
    27: "iliac_artery_right",
    28: "iliac_vena_left",
    29: "iliac_vena_right",
    30: "humerus_left",
    31: "humerus_right",
    32: "scapula_left",
    33: "scapula_right",
    34: "clavicula_left",
    35: "clavicula_right",
    36: "femur_left",
    37: "femur_right",
    38: "hip_left",
    39: "hip_right",
    40: "gluteus_maximus_left",
    41: "gluteus_maximus_right",
    42: "gluteus_medius_left",
    43: "gluteus_medius_right",
    44: "gluteus_minimus_left",
    45: "gluteus_minimus_right",
    46: "autochthon_left",
    47: "autochthon_right",
    48: "iliopsoas_left",
    49: "iliopsoas_right",
    50: "brain",
}


def draw_mask(array: np.ndarray, mask=None, name='ROI', contrast_limits=None) -> np.ndarray:
    """
    Open a napari viewer to manually define a binary mask on a 3D array.
    
    Parameters
    ----------
    array : np.ndarray
        3D array (e.g., shape (z, y, x)) to display in napari.
    mask : np.ndarray
        3D mask to use as basis for editing. If none is provided, the 
        mask is drawn from scratch.
    name : str
        Name of the mask
    
    Returns
    -------
    mask : np.ndarray
        3D binary mask (same shape as input).
        Voxels painted in napari are 1, others are 0.
    """

    # Ensure it's a numpy array
    array = np.asarray(array).T

    # Initialize mask
    if mask is None:
        mask = np.zeros_like(array, dtype=np.uint8)
    else:
        mask = np.asarray(mask).T

    # Start the napari viewer
    viewer = napari.Viewer()
    viewer.add_image(array, name='Data', blending="additive", contrast_limits=contrast_limits)

    # Add an empty labels layer for drawing
    labels_layer = viewer.add_labels(mask, name=name)

    # Set brush tool as active and radius = 6
    viewer.layers.selection.active = labels_layer
    labels_layer.mode = 'paint'
    labels_layer.brush_size = 6

    # Start the interactive GUI
    napari.run()

    # After closing napari, extract the mask
    mask = (labels_layer.data > 0).astype(np.uint16)

    return mask.T



def harmonize_data():

    # If the outputs exists, delete them first
    vfa_series = DATA + ['3D-VFA']
    dce_series = DATA + ['3D-DCE']
    db.delete(vfa_series, not_exists_ok=True)
    db.delete(dce_series, not_exists_ok=True)

    # Merge VFA into 1 series to simplify analysis
    # Write the correct flip angle in each series
    for fa in [2,4,10,30]:
        fa_series = db.series(datapath, contains=f'flip {fa}')[0]
        fa_series_copy = db.copy(fa_series)
        db.edit(fa_series_copy, {'FlipAngle':fa})
        db.move(fa_series_copy, vfa_series) # bug not removing empty folders

    # Merge DCE into one single series
    dce = db.series(datapath, contains='WITH CONTRAST')
    [db.copy(phase, dce_series) for phase in dce]
    # Store timing info in acquisition time in seconds (miblab default)
    acq_time = db.values(dce_series, 'TriggerTime', dims=['SliceLocation', 'TriggerTime'])
    db.edit(dce_series, {'AcquisitionTime': acq_time/1000}, dims=['SliceLocation', 'TriggerTime'])


def compute_descriptive_maps():

    # Compute maps of descriptive parameters and save
    dce_series = DATA + ['3D-DCE']
    dce_data = db.volume(dce_series, 'AcquisitionTime')
    dce_maps = dc.describe(dce_data.values)
    for par in dce_maps:
        par_vol = (dce_maps[par], dce_data.affine)
        par_series = MAPS + [par]
        db.delete(par_series, not_exists_ok=True) 
        db.write_volume(par_vol, par_series, ref=dce_series)


def autosegment_lungs():

    # --- 1. Draw lungs on baseline image
    ref_series = MAPS + ['Sb']
    ref_vol = db.volume(ref_series) 

    # Create label and save
    label_vol = miblab.totseg(ref_vol, cutoff=0.01, task='total_mr', device='cpu')
    total_mr_series = MASKS + ['total_mr']
    db.delete(total_mr_series, not_exists_ok=True) 
    db.write_volume(label_vol, total_mr_series, ref=ref_series)

    # Create individual masks for key structures
    for label_value in [10,11]:
        mask_arr = np.where(label_vol.values==label_value, 1, 0).astype(np.int16)
        mask_series = MASKS + [TOTAL_MR[label_value]]
        db.delete(mask_series, not_exists_ok=True) 
        db.write_volume((mask_arr, label_vol.affine), mask_series, ref=ref_series)

    # # --- 2. Draw liver/kidneys/vessels on maximum enhancement image
    # ref_series = MAPS + ['SEmax']
    # ref_vol = db.volume(ref_series) 

    # # Create label and save
    # label_vol = miblab.totseg(ref_vol, cutoff=0.01, task='total_mr', device='cpu')
    # db.write_volume(label_vol, MASKS + ['total_mr_SEmax'], ref=ref_series)

    # # Create individual masks for key structures
    # for label_value in [2,3,5,22,23,24,25]:
    #     mask_arr = np.where(label_vol.values==label_value, 1, 0).astype(np.int16)
    #     db.write_volume((mask_arr, label_vol.affine), MASKS + [TOTAL_MR[label_value]], ref=ref_series)


def edit_autosegmented_lungs():

    # Edit lungs on baseline image
    ref_series = MAPS + ['Sb']
    ref_vol = db.volume(ref_series) 

    # Edit lung ROIs
    masks_to_edit = ['lung_left', 'lung_right']
    for mask in masks_to_edit:
        mask_array = db.volume(MASKS + [mask]).values
        mask_array = draw_mask(ref_vol.values, mask=mask_array, name=mask)
        mask_vol = (mask_array, ref_vol.affine)
        mask_series = MASKS + [f"{mask}_edited"]
        db.delete(mask_series, not_exists_ok=True)
        db.write_volume(mask_vol, mask_series, ref=ref_series)


def draw_arterial_inputs():

    # Draw right heart and pulmonary artery on SEmax
    ref_series = MAPS + ['SEmax']
    ref_vol = db.volume(ref_series) 

    # Draw Right heart
    roi = 'input'
    mask_array = draw_mask(ref_vol.values, name=roi)  
    mask_vol = (mask_array, ref_vol.affine)
    mask_series = MASKS + [roi]
    db.delete(mask_series, not_exists_ok=True)
    db.write_volume(mask_vol, mask_series, ref=ref_series)

    # Draw AIF
    roi = 'pulmonary_artery'
    mask_array = draw_mask(ref_vol.values, name=roi)  
    mask_vol = (mask_array, ref_vol.affine)
    mask_series = MASKS + [roi]
    db.delete(mask_series, not_exists_ok=True)
    db.write_volume(mask_vol, MASKS + [roi], ref=ref_series)



def vfa_motion_correction():

    moco_series = MAPS + ['3D-VFA-MOCO']
    fit_series = MAPS + ['3D-VFA-MOCO-FIT']
    db.delete(moco_series, not_exists_ok=True)
    db.delete(fit_series, not_exists_ok=True)

    # Get data
    vfa_series = DATA + ['3D-VFA']
    vfa_vol = db.volume(vfa_series, 'FlipAngle')
    
    # Perform motion correction
    coreg, fit, transfo, pars = mdreg.fit(
        vfa_vol.values, 
        fit_image = {
            'func': mdreg.fit_spgr_vfa_lin,
            'FA': vfa_vol.coords[0], 
            'parallel': False,
            'progress_bar': True,                          
        }, 
        fit_coreg = {
            'package': 'ants',
            'type_of_transform': 'SyNOnly',
        }, 
        maxit = 5,       
        verbose = 2,
    )

    # # Compute R1
    # tr_sec = db.unique('RepetitionTime', vfa_series)[0]/1000 
    # R1 = - np.log(pars[...,1]) / tr_sec

    # Save results to DICOM
    vol_coreg = vreg.volume(coreg, vfa_vol.affine, coords=vfa_vol.coords, dims=vfa_vol.dims)
    vol_fit = vreg.volume(fit, vfa_vol.affine, coords=vfa_vol.coords, dims=vfa_vol.dims)
    db.write_volume(vol_coreg, moco_series, ref=vfa_series)
    db.write_volume(vol_fit, fit_series, ref=vfa_series)


def vfa_fit():

    R1_series = MAPS + ['R1_vfa']
    S0_series = MAPS + ['S0_vfa']
    db.delete(R1_series, not_exists_ok=True)
    db.delete(S0_series, not_exists_ok=True)
    
    # Get input data
    vfa_series = MAPS + ['3D-VFA-MOCO']
    vfa_volume = db.volume(vfa_series, 'FlipAngle')
    tr_sec = db.unique('RepetitionTime', vfa_series)[0]/1000 
    vfa_array = vfa_volume.values
    fa_deg = vfa_volume.coords[0]
    S0_max = 10 * np.amax(vfa_array) / np.sin(np.deg2rad(np.min(fa_deg)))
    R1_max = 10 # sec
    bounds = ([0,0], [R1_max, S0_max])
    
    # Compute linear and save results
    R1_array, S0_array = dc.vfa_linear(vfa_array, fa_deg, tr_sec, bounds, verbose=0)
    db.write_volume((R1_array, vfa_volume.affine), R1_series, ref=vfa_series)
    db.write_volume((S0_array, vfa_volume.affine), S0_series, ref=vfa_series)

    # # Compute nonlinear and save results
    # R1_array, S0_array = dc.vfa_nonlinear(vfa_array, fa_deg, tr_sec, bounds, verbose=0)
    # db.write_volume((R1_array, vfa_volume.affine), MAPS + ['R1_vfa_nonlin'], ref=vfa_series)
    # db.write_volume((S0_array, vfa_volume.affine), MAPS + ['S0_vfa_nonlin'], ref=vfa_series)


# def dce_fit():

#     # Get data
#     dce_series = DATA + ['3D-DCE']
#     dce_vol = db.volume(dce_series, 'TriggerTime')

#     n_comp = 8
#     fit, pars = mdreg.fit_pca(dce_vol.values, n_comp)  
#     error = np.linalg.norm(fit-dce_vol.values, axis=-1)

#     # Save results to DICOM
#     vol_comp = vreg.volume(pars, dce_vol.affine, coords=[np.arange(n_comp)], dims=['TriggerTime'])
#     vol_fit = vreg.volume(fit, dce_vol.affine, coords=dce_vol.coords, dims=dce_vol.dims)
#     db.write_volume(vol_comp, MAPS + [f'3D-DCE-PCA-COMP_{n_comp}'], ref=dce_series)
#     db.write_volume(vol_fit, MAPS + [f'3D-DCE-PCA-FIT_{n_comp}'], ref=dce_series)
#     db.write_volume((error, dce_vol.affine), MAPS + [f'3D-DCE-PCA-ERROR_{n_comp}'], ref=dce_series)


def dce_motion_correction():

    moco_series = MAPS + ['3D-DCE-MOCO']
    db.delete(moco_series, not_exists_ok=True)

    # Get data
    dce_series = DATA + ['3D-DCE']
    dce_vol = db.volume(dce_series, 'AcquisitionTime')

    # OPTION 1: DO NOTHING
    db.write_volume(dce_vol, moco_series, ref=dce_series)

    # OPTION 2: MDREG - requires more optimization and evaluation on moving data.

    # aif_mask = db.volume(MASKS + ['input']).values
    # time = dce_vol.coords[0]/1000
    # time = time - time[0]
    # aif_signal = np.mean(dce_vol.values[aif_mask > 0, :], axis=0)

    # # Model fitting options
    # fit_pca = {
    #     'func': mdreg.fit_pca,
    #     'n_components': 8,
    # }
    # fit_2cm = {
    #     'func': mdreg.fit_2cm_lin,
    #     'time': time,
    #     'aif': aif_signal,
    #     'baseline': 1,
    #     'input_corr': True,
    # }
    # fit_deconv = {
    #     'func': mdreg.fit_deconvolution,
    #     'aif': aif_signal,
    #     'n0': 1,
    #     'tol': 0.2,
    # }

    # # Coregistration options
    # fit_ants = {
    #     'package': 'ants',
    #     'type_of_transform': 'SyNOnly',
    #     'parallel': False,
    #     'progress_bar': True,  
    # }
    # fit_skimage = {
    #     'package': 'skimage',
    #     'attachment': 10,
    #     'parallel': False,
    #     'progress_bar': True,  
    # }
    
    # # Perform motion correction
    # coreg, fit, _, _ = mdreg.fit(
    #     dce_vol.values, fit_image=fit_deconv, fit_coreg=fit_ants, 
    #     maxit = 5, verbose = 2,
    # )

    # # Save results to DICOM
    # vol_coreg = vreg.volume(coreg, dce_vol.affine, coords=dce_vol.coords, dims=dce_vol.dims)
    # vol_fit = vreg.volume(fit, dce_vol.affine, coords=dce_vol.coords, dims=dce_vol.dims)
    # db.write_volume(vol_coreg, MAPS + ['3D-DCE-MOCO'], ref=dce_series)
    # db.write_volume(vol_fit, MAPS + ['3D-DCE-MOCO-FIT'], ref=dce_series)
      

def align_vfa_with_dce():

    S0_series = MAPS + ['S0_vfa_on_dce']
    R1_series = MAPS + ['R1_vfa_on_dce']
    db.delete(S0_series, not_exists_ok=True)
    db.delete(R1_series, not_exists_ok=True)

    # Slice R1 and S0 on DCE
    dce_Sb_vol = db.volume(MAPS + ['Sb'])
    vfa_S0_vol = db.volume(MAPS + ['S0_vfa'])
    vfa_R1_vol = db.volume(MAPS + ['R1_vfa'])
    vfa_S0_on_dce_vol = vfa_S0_vol.slice_like(dce_Sb_vol)
    vfa_R1_on_dce_vol = vfa_R1_vol.slice_like(dce_Sb_vol)

    # Compute DCE S0 with R1 map
    dce_series = MAPS + ['3D-DCE-MOCO']
    pars = db.unique(['FlipAngle','RepetitionTime'], dce_series)
    Sref = dc.signal_ss(1, vfa_R1_on_dce_vol.values, pars['RepetitionTime'][0]/1000, pars['FlipAngle'][0])
    dce_S0 = np.divide(dce_Sb_vol.values, Sref, out=np.zeros_like(Sref, dtype=float), where=Sref!=0)

    # Rescale vfa S0 to dce S0 to correct for recalibration between vfa and dce
    liver_mask = db.volume(MASKS + ['total_mr']).values == 5
    liver_vfa_S0 = np.mean(vfa_S0_on_dce_vol.values[liver_mask])
    liver_dce_S0 = np.mean(dce_S0[liver_mask])
    vfa_S0_on_dce_scaled = vfa_S0_on_dce_vol.values * liver_dce_S0 / liver_vfa_S0

    # Clip S0 maps at 10 x the liver values
    S0_max = 100 * liver_dce_S0
    dce_S0[dce_S0 >  S0_max] = S0_max
    vfa_S0_on_dce_scaled[vfa_S0_on_dce_scaled > S0_max] = S0_max

    # Save S0 and R1 maps
    # db.write_volume((dce_S0, dce_Sb_vol.affine), MAPS + ['S0_dce'], ref=dce_series)
    db.write_volume((vfa_S0_on_dce_scaled, dce_Sb_vol.affine), S0_series, ref=dce_series)
    db.write_volume((vfa_R1_on_dce_vol.values, dce_Sb_vol.affine), R1_series, ref=dce_series)


def roi_analysis_model_free():

    dce_series = MAPS + ['3D-DCE-MOCO']
    dce_data = db.volume(dce_series, 'AcquisitionTime')
    aif_mask = db.volume(MASKS + ['pulmonary_artery']).values
    R10_map = db.volume(MAPS + ['R1_vfa_on_dce']).values
    S0_map = db.volume(MAPS + ['S0_vfa_on_dce']).values

    time = dce_data.coords[0]
    time = time - time[0]
    aif_signal = np.mean(dce_data.values[aif_mask > 0, :], axis=0)
    pars = db.unique(['FlipAngle','RepetitionTime'], dce_series)

    for roi in ['lung_left_edited', 'lung_right_edited']:

        lung_mask = db.volume(MASKS + [roi]).values
        lung_signal = np.mean(dce_data.values[lung_mask > 0, :], axis=0)

        lung = dc.TissueLS(
            sequence = 'SS', 
            dt = time[1], 
            TR = pars['RepetitionTime'][0]/1000,
            FA = pars['FlipAngle'][0],
            r1 = dc.relaxivity(3, 'blood', 'gadobutrol'), 
            R10 = np.mean(R10_map[lung_mask > 0]),
            S0 = np.mean(S0_map[lung_mask > 0]),
            R10a = np.mean(R10_map[aif_mask > 0]),
            S0a = np.mean(S0_map[aif_mask > 0]),
        )
        lung.train(lung_signal, aif_signal, tol=0.15, init_s0=False)
        lung.plot(lung_signal, round_to=3, show=False, fname=os.path.join(imagepath, f'model_free_{roi}.png'))


def roi_analysis_model_based():

    dce_series = MAPS + ['3D-DCE-MOCO']
    dce_data = db.volume(dce_series, 'AcquisitionTime')
    aif_mask = db.volume(MASKS + ['pulmonary_artery']).values
    R10_map = db.volume(MAPS + ['R1_vfa_on_dce']).values
    S0_map = db.volume(MAPS + ['S0_vfa_on_dce']).values

    time = dce_data.coords[0]
    time = time - time[0]
    aif_signal = np.mean(dce_data.values[aif_mask > 0, :], axis=0)
    pars = db.unique(['FlipAngle','RepetitionTime'], dce_series)

    for roi in ['lung_left_edited', 'lung_right_edited']:

        lung_mask = db.volume(MASKS + [roi]).values
        lung_signal = np.mean(dce_data.values[lung_mask > 0, :], axis=0)

        # Perform the analysis
        lung = dc.Tissue(
            aif = aif_signal, 
            t = time, 
            kinetics = 'NXP', 
            TR = pars['RepetitionTime'][0]/1000,
            FA = pars['FlipAngle'][0],
            r1 = dc.relaxivity(3, 'blood', 'gadobutrol'), 
            R10 = np.mean(R10_map[lung_mask > 0]),
            S0 = np.mean(S0_map[lung_mask > 0]),
            R10a = np.mean(R10_map[aif_mask > 0]),
            S0a = np.mean(S0_map[aif_mask > 0]),
            n0 = 5,
            free = {'Fb':[0,1], 'vb':[0,1]}, 
        )
        lung.train(time, lung_signal, init_s0=False)
        lung.plot(time, lung_signal, round_to=3, show=False, fname=os.path.join(imagepath, f'model_based_{roi}.png'))


def pixel_analysis_model_free():

    # If the outputs exists, delete them first
    Fb_series = MAPS + ['Fb_free']
    ve_series = MAPS + ['ve_free']
    db.delete(Fb_series, not_exists_ok=True)
    db.delete(ve_series, not_exists_ok=True)

    # Read data volumes
    dce_series = MAPS + ['3D-DCE-MOCO']
    dce_data = db.volume(dce_series, 'AcquisitionTime')
    aif_mask = db.volume(MASKS + ['pulmonary_artery']).values > 0
    R10_map = db.volume(MAPS + ['R1_vfa_on_dce']).values
    S0_map = db.volume(MAPS + ['S0_vfa_on_dce']).values

    # Read signals and sequence parameters
    time = dce_data.coords[0]
    aif_signal = np.mean(dce_data.values[aif_mask, :], axis=0)
    pars = db.unique(['FlipAngle','RepetitionTime'], dce_series)

    # Train a tissue model
    body = dc.TissueLSArray(
        dce_data.shape[:3],
        sequence = 'SS', 
        dt = time[1] - time[0], 
        TR = pars['RepetitionTime'][0]/1000,
        FA = pars['FlipAngle'][0],
        r1 = dc.relaxivity(3, 'blood', 'gadobutrol'), 
        R10 = R10_map,  
        S0 = S0_map,
        R10a = np.mean(R10_map[aif_mask]),
        S0a = np.mean(S0_map[aif_mask]),
    )
    body.train(dce_data.values, aif_signal, tol=0.15, init_s0=False)
    
    # vol_conc = vreg.volume(body.predict_conc(), dce_data.affine, coords=dce_data.coords, dims=dce_data.dims)
    # db.write_volume(vol_conc, MAPS + ['C'+ext], ref=dce_series)
    pars = body.params()
    db.write_volume((pars['Fb'], dce_data.affine), Fb_series, ref=dce_series)
    db.write_volume((pars['ve'], dce_data.affine), ve_series, ref=dce_series)
    #db.write_volume((pars['S0'], dce_data.affine), MAPS + ['S0_free'+ext], ref=dce_series)

    lung_right_mask = db.volume(MASKS + ['lung_right']) 
    lung_left_mask = db.volume(MASKS + ['lung_left'])
    lung_mask = (lung_right_mask.values==1) | (lung_left_mask.values==1)
    vmin = {'Fb':0, 've':0, 'background':0}
    vmax = {'Fb':0.5, 've':1.0, 'background':np.percentile(S0_map, 99)}
    body.plot_overlay(lung_mask, vmin=vmin, vmax=vmax, show=False, fname=os.path.join(imagepath, f'model_free.png'))


def pixel_analysis_model_based():

    Fb_series = MAPS + ['Fb_model']
    vb_series = MAPS + ['vb_model']
    db.delete(Fb_series, not_exists_ok=True)
    db.delete(vb_series, not_exists_ok=True)

    # Read data volumes
    dce_series = MAPS + ['3D-DCE-MOCO']
    dce_data = db.volume(dce_series, 'AcquisitionTime')
    aif_mask = db.volume(MASKS + ['pulmonary_artery']).values > 0
    R10_map = db.volume(MAPS + ['R1_vfa_on_dce']).values
    S0_map = db.volume(MAPS + ['S0_vfa_on_dce']).values
    lung_right_mask = db.volume(MASKS + ['lung_right']).values 
    lung_left_mask = db.volume(MASKS + ['lung_left']).values

    # Get signals, sequence parameters and masks
    time = dce_data.coords[0]
    time = time - time[0]
    aif_signal = np.mean(dce_data.values[aif_mask, :], axis=0)
    pars = db.unique(['FlipAngle','RepetitionTime'], dce_series)
    lung_mask = (lung_right_mask==1) | (lung_left_mask==1)
    dce_lung = dce_data.values[lung_mask, :]

    # # Read
    # vmin = {'Fb':0, 've':0, 'background':0}
    # vmax = {'Fb':0.5, 've':1.0, 'background':np.percentile(S0_map, 99)}
    # Fb = db.volume(MAPS + ['Fb_model']).values
    # ve = db.volume(MAPS + ['ve_model']).values
    # lung = dc.TissueArray(dce_data.shape[:3], kinetics='FX', sequence='SS', Fb=Fb, ve=ve, R10=R10_map, S0=S0_map).load(os.path.join(checkpointspath, f'model_3d'))
    # lung.plot_overlay(dce_data.values, lung_mask, vmin=vmin, vmax=vmax, show=False, fname=os.path.join(imagepath, f'model_based.png'))

    # return

    model_pars = {
        'aif': aif_signal, 
        't': time, 
        'TR': pars['RepetitionTime'][0]/1000,
        'FA': pars['FlipAngle'][0],
        'r1': dc.relaxivity(3, 'blood', 'gadobutrol'), 
        'R10': R10_map[lung_mask], 
        'S0': S0_map[lung_mask],
        'R10a': np.mean(R10_map[aif_mask]),
        'S0a': np.mean(S0_map[aif_mask]),
        'n0': 3,
        'parallel': False,
        'verbose': 1,   
        'free': {'Fb':[0,1], 'vb':[0,1]},
    }
   
    # Train a tissue model (scale by max)
    lung = dc.TissueArray(dce_lung.shape[:1], kinetics='NXP', sequence='SS', **model_pars)
    lung.train(time, dce_lung, init_s0=False)
    lung.save(os.path.join(checkpointspath, f'model_flat'))

    # Create maps
    vb = np.zeros(dce_data.shape[:3])
    Fb = np.zeros(dce_data.shape[:3])

    vb[lung_mask] = lung.params('vb')
    Fb[lung_mask] = lung.params('Fb')

    # Save results
    db.write_volume((vb, dce_data.affine), vb_series, ref=dce_series)
    db.write_volume((Fb, dce_data.affine), Fb_series, ref=dce_series)

    # Load again with 3D maps
    vmin = {'Fb':0, 'vb':0, 'background':0}
    vmax = {'Fb':0.5, 'vb':1.0, 'background':np.percentile(S0_map, 99)}
    model_pars['R10'] = R10_map
    model_pars['S0'] = S0_map
    lung = dc.TissueArray(dce_data.shape[:3], kinetics='NXP', sequence='SS', Fb=Fb, vb=vb, **model_pars)
    lung.plot_overlay(dce_data.values, lung_mask, vmin=vmin, vmax=vmax, show=False, fname=os.path.join(imagepath, f'model_based.png'))
    lung.save(os.path.join(checkpointspath, f'model_3d'))


def view_maps():

    # img, mask = 'S0_vfa_on_dce', "lung_left"
    # vol = db.volume(MAPS + [img])
    # mask_vol = db.volume(MASKS + [mask])
    # draw_mask(vol.values, mask_vol.values, name=mask, contrast_limits=[0, 5000])

    map = 'S0_dce'
    vol = db.volume(MAPS + [map])
    viewer = napari.Viewer()
    viewer.add_image(vol.values.T, name=map, contrast_limits=[0, 5000])
    napari.run()

    map = 'S0_vfa_on_dce'
    vol = db.volume(MAPS + [map])
    viewer = napari.Viewer()
    viewer.add_image(vol.values.T, name=map, contrast_limits=[0, 5000])
    napari.run()



if __name__=='__main__':
    # harmonize_data()
    # compute_descriptive_maps()
    # autosegment_lungs()
    # edit_autosegmented_lungs()
    # draw_arterial_inputs()
    # vfa_motion_correction()
    # vfa_fit()
    # dce_motion_correction()
    # align_vfa_with_dce()
    # roi_analysis_model_based()
    # roi_analysis_model_free()
    pixel_analysis_model_free()
    # pixel_analysis_model_based()
    # view_maps()
    