import glob
import os
import numpy as np
from torchvision.datasets import ImageFolder
import pandas as pd


def _process_dir(dir_path, is_train=False):
    img_path = ImageFolder(dir_path).imgs
    accumulated_info_list = []
    for path, v in img_path:
        filename = path.split('/')[-1]
        if not is_train:
            label = filename.split('_')[0]
            camera = filename.split('c')[1]
            if label[0:10]=='0000others':
                label_id = -1000
            else:
                label_id = int(label)
            camera_id = int(camera[0])
        else:
            # Dummy label and camera ids for train images as this is a weakly supervised dataset
            label_id = -4e7
            camera_id = -4e7

        accumulated_info = {
            "image_file_path": path,
            "identity_ID": label_id,
            "camera_ID": camera_id
        }
        accumulated_info_list.append(accumulated_info)
    return accumulated_info_list



def _load_accumulated_info(root_folder_path,
                           dataset_folder_name="sysu-30k-release",
                           image_folder_name="sysu_test_set_all", 
                           subfolder_name = "gallery"):
    """SYSU-30k.
    Reference:
        SYSU-30k: Weakly Supervised Person Re-ID: Differentiable Graphical Learning and A New Benchmark

    URL: `<https://github.com/wanggrun/SYSU-30k>`

    Dataset statistics:
        SYSU-30k contains 30k categories of persons, which is about 20 times larger than CUHK03 (1.3k categories) 
        and Market1501 (1.5k categories), and 30 times larger than ImageNet (1k categories). SYSU-30k contains 
        29,606,918 images. Moreover, SYSU-30k provides not only a large platform for the weakly supervised ReID problem 
        but also a more challenging test set that is consistent with the realistic setting for standard evaluation. 


        Comparision with existing datasets
        -----------------------------------------------------------------------------------------------------------------------------------------
        |   Dataset	    CUHK03	    Market-1501	    Duke	    MSMT17	        CUHK01	    PRID	    VIPeR	    CAVIAR	    SYSU-30k        |
        -----------------------------------------------------------------------------------------------------------------------------------------
        |   Categories	1,467	    1,501	        1,812	    4,101	        971	        934	        632	        72	        30,508          |
        -----------------------------------------------------------------------------------------------------------------------------------------
        |   Scene	    Indoor	    Outdoor	        Outdoor	    Indoor,Outdoor	Indoor	    Outdoor	    Outdoor	    Indoor	    Indoor,Outdoor  |
        -----------------------------------------------------------------------------------------------------------------------------------------
        |   Annotation	Strong	    Strong	        Strong	    Strong	        Strong	    Strong	    Strong	    Strong	    Weak            |
        -----------------------------------------------------------------------------------------------------------------------------------------
        |   Cameras	    2	        6	            8	        15	            10	        2	        2	        2	        Countless       |
        -----------------------------------------------------------------------------------------------------------------------------------------
        |   Images	    28,192	    32,668	        36,411	    126,441	        3,884	    1,134	    1,264	    610	        29,606,918      |
        -----------------------------------------------------------------------------------------------------------------------------------------


        Comparision with ImageNet-1k
        --------------------------------------------
        | Dataset	   | ImageNet-1k  |  SYSU-30k  |
        --------------------------------------------
        | Categories   | 1,000	      |  30,508    |
        --------------------------------------------
        | Images	   | 1,280,000	  |  29,606,918|
        --------------------------------------------
        | Annotation   | Strong	      |  Weak      |
        --------------------------------------------
    """
    dataset_folder_path = os.path.join(root_folder_path, dataset_folder_name)

    if subfolder_name is not None:
        image_folder_path = os.path.join(dataset_folder_path, image_folder_name, subfolder_name)
    else:
        image_folder_path = os.path.join(dataset_folder_path, image_folder_name)

    if image_folder_name == "sysu_train_set_all":
        accumulated_info_list = _process_dir(image_folder_path, is_train=True)
    else:
        accumulated_info_list = _process_dir(image_folder_path, is_train=False)

    # Convert list to data frame
    accumulated_info_dataframe = pd.DataFrame(accumulated_info_list)
    return accumulated_info_dataframe


def load_sysu30k(root_folder_path):
    train_and_valid_accumulated_info_dataframe = _load_accumulated_info(
        root_folder_path=root_folder_path,
        image_folder_name="sysu_train_set_all", subfolder_name=None)
    test_query_accumulated_info_dataframe = _load_accumulated_info(
        root_folder_path=root_folder_path, image_folder_name="sysu_test_set_all", subfolder_name="query")
    test_gallery_accumulated_info_dataframe = _load_accumulated_info(
        root_folder_path=root_folder_path, image_folder_name="sysu_test_set_all", subfolder_name="gallery")
    return train_and_valid_accumulated_info_dataframe, test_query_accumulated_info_dataframe, test_gallery_accumulated_info_dataframe