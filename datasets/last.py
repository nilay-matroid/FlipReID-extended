import glob
import os
import numpy as np

import pandas as pd


def _get_pid2label(dir_path):
    img_paths = glob.glob(os.path.join(dir_path, '*/*.jpg'))
    pid_container = set()
    for img_path in img_paths:
        pid = int(os.path.basename(img_path).split('_')[0])
        pid_container.add(pid)
    pid_container = np.sort(list(pid_container))
    pid2label = {pid: label for label, pid in enumerate(pid_container)}
    return pid2label


def _process_dir(dir_path, pid2label=None, relabel=False, recam=0):
    if 'query' in dir_path:
        img_paths = glob.glob(os.path.join(dir_path, '*.jpg'))
    else:
        img_paths = glob.glob(os.path.join(dir_path, '*/*.jpg'))
    img_paths = sorted(img_paths)
    accumulated_info_list = []
    for ii, img_path in enumerate(img_paths):
        pid = int(os.path.basename(img_path).split('_')[0])
        camid = int(recam + ii)
        if relabel and pid2label is not None:
            pid = pid2label[pid]
        accumulated_info = {
            "image_file_path": img_path,
            "identity_ID": pid,
            "camera_ID": camid
        }
        accumulated_info_list.append(accumulated_info)
    return accumulated_info_list


def _load_accumulated_info(root_folder_path,
                           dataset_folder_name="last",
                           image_folder_name="test", 
                           subfolder_name = "gallery", recam=0):
    """LaST.

    Reference:
        LaST: Large-Scale Spatio-Temporal Person Re-identification

    URL: `<https://github.com/shuxjweb/last#last-large-scale-spatio-temporal-person-re-identification>`_

    Dataset statistics:
        LaST is a large-scale dataset with more than 228k pedestrian images. 
        It is used to study the scenario that pedestrians have a large activity scope and time span. 
        Although collected from movies, we have selected suitable frames and labeled them as carefully as possible. 
        Besides the identity label, we also labeled the clothes of pedestrians in the training set.
        
        Train: 5000 identities and 71,248 images.
        Val: 56 identities and 21,379 images.
        Test: 5806 identities and 135,529 images.
        --------------------------------------
        subset         | # ids     | # images
        --------------------------------------
        train          |  5000     |    71248
        query          |    56     |      100
        gallery        |    56     |    21279
        query_test     |  5805     |    10176
        gallery_test   |  5806     |   125353
    """
    dataset_folder_path = os.path.join(root_folder_path, dataset_folder_name)

    if subfolder_name is not None:
        image_folder_path = os.path.join(dataset_folder_path, image_folder_name, subfolder_name)
    else:
        image_folder_path = os.path.join(dataset_folder_path, image_folder_name)

    if image_folder_name == "train":
        pid2label = _get_pid2label(image_folder_path)
        accumulated_info_list = _process_dir(image_folder_path, pid2label=pid2label, relabel=True)
    else:
        accumulated_info_list = _process_dir(image_folder_path, relabel=False, recam=recam)

    # Convert list to data frame
    accumulated_info_dataframe = pd.DataFrame(accumulated_info_list)
    return accumulated_info_dataframe


def load_LaST(root_folder_path, use_eval_set=False):
    test_folder = "test"
    if use_eval_set:
        test_folder = "val"
    train_and_valid_accumulated_info_dataframe = _load_accumulated_info(
        root_folder_path=root_folder_path,
        image_folder_name="train", subfolder_name=None)
    test_query_accumulated_info_dataframe = _load_accumulated_info(
        root_folder_path=root_folder_path, image_folder_name=test_folder, subfolder_name="query")
    test_gallery_accumulated_info_dataframe = _load_accumulated_info(
        root_folder_path=root_folder_path,
        image_folder_name=test_folder, subfolder_name="gallery", recam=len(test_query_accumulated_info_dataframe))
    return train_and_valid_accumulated_info_dataframe, test_query_accumulated_info_dataframe, test_gallery_accumulated_info_dataframe
