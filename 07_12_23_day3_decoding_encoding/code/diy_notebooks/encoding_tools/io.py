import os
import numpy as np

def download_neural_data(data_type, destination = None):
    '''Function to automatically download either miniscope or widefield data from
    the google drive

    Parameters
    ----------
    data_type: string, either "widefield" or "miniscope"
    destination: string, desired folder to store the data, default = download_neural_data

    Returns
    -------
    data_file: string, the path to the data file for loading

    ----------------------------------------------------------------------------
    '''
    import gdown #Make sure to install this first via: pip install gdown
    import os

    if data_type == "miniscope":
        url = "https://drive.google.com/file/d/1JT0TcbWDKsB90CRMy0XX8dCSaO7Dxvi8/view?usp=drive_link"
        fname = "miniscope_data.npy"
    elif data_type == "widefield":
        url = "https://drive.google.com/file/d/1XNCPKY5bRS9QtvY1aj982CjaCkMgeOJt/view?usp=drive_link"
        fname = "widefield_data.mat" #Some renaming here to simplify

    if destination is None:
        parent = os.path.split(os.getcwd())[0] #The parent directory
        destination = os.path.join(parent, "DataSAI_data_folder")
        if not os.path.isdir(destination):
            os.makedirs(destination)
    data_file = os.path.join(destination, fname)

    if not os.path.exists(data_file):
        gdown.download(url, data_file, quiet=False, fuzzy =True)
    return data_file

def load_miniscope(data_path):
    data = np.load(data_path, allow_pickle=True).tolist()

    design_matrix = data['design_matrix']
    Y_raw_fluorescence = data['Y_raw_fluorescence']
    neuron_footprints = data['neuron_footprints']
    timepoints_per_trial = data['timepoints_per_trial']
    frame_rate = data['frame_rate']
    aligned_segment_start = data['aligned_segment_start']
    return design_matrix, Y_raw_fluorescence, neuron_footprints, timepoints_per_trial, frame_rate, aligned_segment_start
