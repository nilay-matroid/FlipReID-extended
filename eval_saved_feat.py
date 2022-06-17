from sklearn.metrics import pairwise_distances
from evaluation.metrics import compute_CMC_mAP
from evaluation.post_processing.re_ranking_ranklist import re_ranking
import argparse
import numpy as np
parser = argparse.ArgumentParser(description='Eval saved features')
parser.add_argument('--use_re_ranking', action='store_true', help='Do re_ranking')
parser.add_argument('--saved_feat_path', default='',type=str, help='Path at which features, ids and camids are saved for both query and gallery')
parser.add_argument('--max_rank', default=20, type=int, help='max rank to evaluate stats')
parser.add_argument('--metric', default='cosine',type=str, help='metric for computing distance array')

def compute_distance_matrix(query_image_features, gallery_image_features, metric, use_re_ranking):
    # Compute the distance matrix
    query_gallery_distance = pairwise_distances(query_image_features,
                                                gallery_image_features,
                                                metric=metric)
    distance_matrix = query_gallery_distance

    # Use the re-ranking method
    if use_re_ranking:
        query_query_distance = pairwise_distances(query_image_features,
                                                    query_image_features,
                                                    metric=metric)
        gallery_gallery_distance = pairwise_distances(
            gallery_image_features, gallery_image_features, metric=metric)
        distance_matrix = re_ranking(query_gallery_distance,
                                        query_query_distance,
                                        gallery_gallery_distance)

    return distance_matrix

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

    rank_list = [1, 5, 10, 20]
    split_name = "test"

    # Compute distance matrix
    distance_matrix = compute_distance_matrix(query_image_features_array, gallery_image_features_array,\
         args.metric, args.use_re_ranking)

    # Compute the CMC and mAP scores
    CMC_score_array, mAP_score = compute_CMC_mAP(
        distmat=distance_matrix,
        q_pids=query_identity_ID_array,
        g_pids=gallery_identity_ID_array,
        q_camids=query_camera_ID_array,
        g_camids=gallery_camera_ID_array,
        max_rank=args.max_rank)

    # print CMC and mAP scores
    
    for rank in rank_list:
        print(f"{split_name}_{args.metric}_{args.use_re_ranking}_rank_to_accuracy_dict rank-{rank} accuracy {CMC_score_array[rank - 1]}")

    print(f"{split_name}_{args.metric}_{args.use_re_ranking}_mAP_score {mAP_score}")