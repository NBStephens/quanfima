import os
import sys
import pathlib
import numpy as np
from skimage import filters
from quanfima import morphology as mrph
#from quanfima import visualization as vis
from quanfima import utils

input_dir = pathlib.Path(r"D:\Desktop\quanfima\data")
input_file = "polymer3d_8bit_128x128x128.raw"

inputImage = pathlib.Path(input_dir).joinpath(input_file)
output_dir = pathlib.Path(input_dir).joinpath("results")

data = np.memmap(inputImage, shape=(128,128,128), dtype=np.uint8, mode='r')

data_seg = np.zeros_like(data, dtype=np.uint8)
for i in range(data_seg.shape[0]):
  th_val = filters.threshold_otsu(data[i])
  data_seg[i] = (data[i] > th_val).astype(np.uint8)

# estimate porosity
pr = mrph.calc_porosity(data_seg)
for k,v in pr.items():
  print(f'Porosity ({k}): {v}')

# prepare data and analyze fibers
pdata, pskel, pskel_thick = utils.prepare_data(data_seg)

oprops = mrph.estimate_tensor(name="test",
                              skel=pskel,
                              data=pskel_thick,
                              window_size=64,
                              output_dir=output_dir,
                              sigma=0.025,
                              make_output=True,
                              original=False)

oprops =  mrph.estimate_tensor_parallel(name='polymer_orientation_w32',
                                        skel=pskel,
                                        data=pskel_thick,
                                        window_size=32,
                                        output_dir=output_dir,
                                        sigma=0.025,
                                        make_output=True,
                                        n_processes=8,
                                        original=False)

odata = np.load(oprops['output_path'], allow_pickle=True).item()
lat, azth, skel = odata['lat'], odata['azth'], odata['skeleton']

dprops = mrph.estimate_diameter_single_run(name='polymer_diameter',
                                           output_dir=output_dir,
                                           data=pdata,
                                           skel=skel,
                                           lat_data=lat,
                                           azth_data=azth,
                                           n_scan_angles=32,
                                           make_output=True)

dmtr = np.load(dprops['output_path'], allow_pickle=True).item()['diameter']

# plot results
vis.plot_3d_orientation_map('polymer_w32', lat, azth,
                            output_dir='../../data/results',
                            camera_azth=40.47,
                            camera_elev=32.5,
                            camera_fov=35.0,
                            camera_loc=(40.85, 46.32, 28.85),
                            camera_zoom=0.005124)

vis.plot_3d_diameter_map('polymer_w32', dmtr,
                         output_dir='../../data/results',
                         measure_quantity='vox',
                         camera_azth=40.47,
                         camera_elev=32.5,
                         camera_fov=35.0,
                         camera_loc=(40.85, 46.32, 28.85),
                         camera_zoom=0.005124,
                         cb_x_offset=5,
                         width=620)