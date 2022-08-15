import argparse
import numpy as np
import os
parser = argparse.ArgumentParser(description='Eval saved features')
parser.add_argument('--saved_feat_path', default='',type=str, help='Path at which features, ids and camids are saved for both query and gallery')
parser.add_argument('--resave_feat_dir', default='',type=str, help='Directory at which test.npz is saved as feat.npy, pid.npy and camid.npy')

if __name__ == "__main__":
    args = parser.parse_args()

    feat_directory = args.saved_feat_path
    saved_dict = np.load(feat_directory)
    query_identity_ID_array = saved_dict['query_identity_ID_array']
    gallery_identity_ID_array = saved_dict['gallery_identity_ID_array']
    query_camera_ID_array = saved_dict['query_camera_ID_array']
    gallery_camera_ID_array = saved_dict['gallery_camera_ID_array']
    query_image_features_array = saved_dict['query_image_features_array']
    gallery_image_features_array = saved_dict['gallery_image_features_array']

    pid = np.array(query_identity_ID_array.tolist() + gallery_identity_ID_array.tolist()).astype(np.float32)
    camid = np.array(query_camera_ID_array.tolist() + gallery_camera_ID_array.tolist()).astype(np.float32)
    feat = np.array(query_image_features_array.tolist() + gallery_image_features_array.tolist()).astype(np.float32)

    np.save(os.path.join(args.resave_feat_dir, "feat.npy"), feat)
    np.save(os.path.join(args.resave_feat_dir, "pid.npy"), pid)
    np.save(os.path.join(args.resave_feat_dir, "camid.npy"), camid)
