import os

import dbdicom as db
import napari

# import numpy as np
# import dcmri as dc

# import matplotlib.pyplot as plt
# import dbdicom as db
# import miblab
# import mdreg
# import vreg


datapath = "C:\\Users\\md1spsx\\Documents\\Data\\case_studies\\kidney_2d_dce"
build_path = os.path.join(os.getcwd(), 'build', 'kidney_2d_dce')

DCE_AORTA = [build_path, 'Anonymous', 'source_data', 'DCE-aorta']
DCE_KIDNEY = [build_path, 'Anonymous', 'source_data', 'DCE-kidney']



def clean_data():
    dce_series = db.series(datapath)[0]

    # Copy a clean series to the build
    dce_series_named = [build_path, 'Anonymous', 'source_data', 'DCE']
    db.copy(dce_series, dce_series_named)

    # Split into aorta and kidney series
    dce_series_split = db.split_series(dce_series_named, 'ImageOrientationPatient')
    for orient, series in dce_series_split:
        if orient == [1,0,0,0,1,0]:
            # The axial slice is through the aorta
            db.move(series, DCE_AORTA)
        else:
            # The other slices are kidney slices
            db.move(series, DCE_KIDNEY)

    # We don't need the original
    db.delete(dce_series_named)



def view_maps():

    arr = db.pixel_data(DCE_KIDNEY, 'AcquisitionTime')
    viewer = napari.Viewer()
    viewer.add_image(arr.T, name=map)
    # viewer.add_image(arr.T, name=map, contrast_limits=[0, 1000])
    napari.run()


if __name__=='__main__':
    # clean_data()
    view_maps()