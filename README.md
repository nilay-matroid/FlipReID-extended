![Python](https://img.shields.io/badge/python-3.8-blue?style=flat-square&logo=python)
![TensorFlow](https://img.shields.io/badge/tensorflow-2.2.3-green?style=flat-square&logo=tensorflow)


# Matroid-FlipReID

## Introduction
This is a modified implementation of the paper "FlipReID: Closing the Gap Between Training and Inference in Person Re-Identification" to train and evaluate the model on other new datasets namely - LaST, SYSU-30k and the old academic dataset - Market1501

## Environment
```bash
wget https://repo.anaconda.com/miniconda/Miniconda3-latest-Linux-x86_64.sh
bash Miniconda3-latest-Linux-x86_64.sh
conda config --set auto_activate_base false
conda create --yes --name TensorFlow2.2 python=3.8
conda activate TensorFlow2.2
conda install --yes cudatoolkit=10.1 cudnn=7.6 -c nvidia
conda install --yes cython matplotlib numpy=1.18 pandas pydot scikit-learn
pip install tensorflow==2.2.3
pip install tf2cv
pip install opencv-python
pip install albumentations --no-binary imgaug,albumentations
```

## Evaluation scripts
- Market1501: 
  - Without reranking
    ```bash
    python3 -u solution.py --dataset_name "Market1501" --backbone_model_name "resnesta50" --pretrained_model_file_path "checkpoints/Market1501_resnesta50_18209984.h5" --use_horizontal_flipping_inside_model --use_horizontal_flipping_in_evaluation --output_folder_path "evaluation_only" --evaluation_only --freeze_backbone_for_N_epochs 0 --testing_size 1.0 --evaluate_testing_every_N_epochs 1 --identity_num_per_batch 4 --image_num_per_identity 2 --save_data_to_disk
    ```

  - With reranking
    ```bash
    python3 -u solution.py --dataset_name "Market1501" --backbone_model_name "resnesta50" --pretrained_model_file_path "checkpoints/Market1501_resnesta50_18209984.h5" --use_horizontal_flipping_inside_model --use_horizontal_flipping_in_evaluation --output_folder_path "evaluation_only" --evaluation_only --freeze_backbone_for_N_epochs 0 --testing_size 1.0 --evaluate_testing_every_N_epochs 1 --identity_num_per_batch 4 --image_num_per_identity 2 --use_re_ranking
    ```

- LaST
  - Eval set
    - Without re-ranking
      ```bash
      python3 -u solution.py --dataset_name "LaST" --backbone_model_name "resnesta50" --pretrained_model_file_path "checkpoints/Market1501_resnesta50_18209984.h5" --use_horizontal_flipping_inside_model --use_horizontal_flipping_in_evaluation --output_folder_path "evaluation_only" --evaluation_only --freeze_backbone_for_N_epochs 0 --testing_size 1.0 --evaluate_testing_every_N_epochs 1 --identity_num_per_batch 4 --image_num_per_identity 1 --special_num_classes 751 --workers 1 --use_eval_set --verbose
      ```

    - With re-ranking
      ```bash
      python3 -u solution.py --dataset_name "LaST" --backbone_model_name "resnesta50" --pretrained_model_file_path "checkpoints/Market1501_resnesta50_18209984.h5" --use_horizontal_flipping_inside_model --use_horizontal_flipping_in_evaluation --output_folder_path "evaluation_only" --evaluation_only --freeze_backbone_for_N_epochs 0 --testing_size 1.0 --evaluate_testing_every_N_epochs 1 --identity_num_per_batch 4 --image_num_per_identity 1 --special_num_classes 751 --workers 1 --use_eval_set --verbose --use_re_ranking
      ```

  - Test set
    - Without re-ranking
      ```bash
      python3 -u solution.py --dataset_name "LaST" --backbone_model_name "resnesta50" --pretrained_model_file_path "checkpoints/Market1501_resnesta50_18209984.h5" --use_horizontal_flipping_inside_model --use_horizontal_flipping_in_evaluation --output_folder_path "evaluation_only" --evaluation_only --freeze_backbone_for_N_epochs 0 --testing_size 1.0 --evaluate_testing_every_N_epochs 1 --identity_num_per_batch 2 --image_num_per_identity 1 --special_num_classes 751 --workers 1 --verbose --save_data_to_disk
      ```
      Same evaluation but using script:
      ```bash
      python3 eval_saved_feat.py --saved_feat_path "evaluation_only/LaST_resnesta50/test.npz"
      ```

    - With re-ranking \
      Same scripts with --use_re_ranking argument \
      Too much memory :(

- SYSU-30k
  - Without re-ranking
    ```bash
    python3 -u solution.py --dataset_name "SYSU-30k" --backbone_model_name "resnesta50" --pretrained_model_file_path "checkpoints/Market1501_resnesta50_18209984.h5" --use_horizontal_flipping_inside_model --use_horizontal_flipping_in_evaluation --output_folder_path "evaluation_only" --evaluation_only --freeze_backbone_for_N_epochs 0 --testing_size 1.0 --evaluate_testing_every_N_epochs 1 --identity_num_per_batch 1 --image_num_per_identity 2 --special_num_classes 751 --workers 1 --verbose --save_data_to_disk --epoch_num 1 --use_custom_eval_batch_size --custom_eval_batch_size 64 --save_data_to_disk
    ```
    Takes too much memory :(

## Training scripts
- LaST \
  Following is the script. But takes too much memory.
  ```bash
  python3 -u solution.py --root_folder_path "/home/ubuntu/Datasets" --dataset_name "LaST" --use_horizontal_flipping_inside_model --nouse_horizontal_flipping_in_evaluation --steps_per_epoch 200 --epoch_num 200 --save_data_to_disk --backbone_model_name "resnesta50" --learning_rate_start 3e-4 --learning_rate_end 3e-4 --learning_rate_base 3e-4 --learning_rate_lower_bound 3e-6 --kernel_regularization_factor 0.0010 --bias_regularization_factor 0.0010 --gamma_regularization_factor 0.0010 --beta_regularization_factor 0.0010 --output_folder_path  "train/last" --verbose --use_eval_set --workers 1 --identity_num_per_batch 4 --image_num_per_identity 4
  ```

  For 2 stage training - Initial freezed backbone and then resuming from it:
  - Initial freezed backbone training
    ```bash
    python3 -u solution.py --root_folder_path "/home/ubuntu/Datasets" --dataset_name "LaST" --use_horizontal_flipping_inside_model --nouse_horizontal_flipping_in_evaluation --steps_per_epoch 200 --epoch_num 200 --save_data_to_disk --backbone_model_name "resnesta50" --learning_rate_start 3e-4 --learning_rate_end 3e-4 --learning_rate_base 3e-4 --learning_rate_lower_bound 3e-6 --kernel_regularization_factor 0.0010 --bias_regularization_factor 0.0010 --gamma_regularization_factor 0.0010 --beta_regularization_factor 0.0010 --output_folder_path  "train/last" --verbose --use_eval_set --workers 1
    ```

  - Resuming from freezed backbone training
    ```bash
    python3 -u solution.py --root_folder_path "../Datasets" --dataset_name "LaST" --use_horizontal_flipping_inside_model --nouse_horizontal_flipping_in_evaluation --steps_per_epoch 200 --epoch_num 200 --backbone_model_name "resnesta50" --learning_rate_start 3e-4 --learning_rate_end 3e-4 --learning_rate_base 3e-4 --learning_rate_lower_bound 3e-6 --kernel_regularization_factor 0.0010 --bias_regularization_factor 0.0010 --gamma_regularization_factor 0.0010 --beta_regularization_factor 0.0010 --output_folder_path  "train/last" --verbose --use_eval_set --workers 1 --resume_after_freezing_backbone_training --identity_num_per_batch 4 --image_num_per_identity 4
    ```

# FlipReID: Closing the Gap Between Training and Inference in Person Re-Identification

[![PWC](https://img.shields.io/endpoint.svg?url=https://paperswithcode.com/badge/flipreid-closing-the-gap-between-training-and/person-re-identification-on-msmt17)](https://paperswithcode.com/sota/person-re-identification-on-msmt17?p=flipreid-closing-the-gap-between-training-and)

[![PWC](https://img.shields.io/endpoint.svg?url=https://paperswithcode.com/badge/flipreid-closing-the-gap-between-training-and/person-re-identification-on-market-1501)](https://paperswithcode.com/sota/person-re-identification-on-market-1501?p=flipreid-closing-the-gap-between-training-and)

[![PWC](https://img.shields.io/endpoint.svg?url=https://paperswithcode.com/badge/flipreid-closing-the-gap-between-training-and/person-re-identification-on-dukemtmc-reid)](https://paperswithcode.com/sota/person-re-identification-on-dukemtmc-reid?p=flipreid-closing-the-gap-between-training-and)

## Overview

Since neural networks are data-hungry, incorporating data augmentation in training is a widely adopted technique that enlarges datasets and improves generalization.
On the other hand, aggregating predictions of multiple augmented samples (i.e., test-time augmentation) could boost performance even further.
In the context of person re-identification models, it is common practice to extract embeddings for both the original images and their horizontally flipped variants.
The final representation is the mean of the aforementioned feature vectors.
However, such scheme results in a gap between training and inference, i.e., the mean feature vectors calculated in inference are not part of the training pipeline.
In this study, we devise the FlipReID structure with the flipping loss to address this issue.
More specifically, models using the FlipReID structure are trained on the original images and the flipped images simultaneously, and incorporating the flipping loss minimizes the mean squared error between feature vectors of corresponding image pairs.
Extensive experiments show that our method brings consistent improvements.
In particular, we set a new record for MSMT17 which is the largest person re-identification dataset.
The source code is available at https://github.com/nixingyang/FlipReID.

## Environment

```bash
wget https://repo.anaconda.com/miniconda/Miniconda3-latest-Linux-x86_64.sh
bash Miniconda3-latest-Linux-x86_64.sh
conda config --set auto_activate_base false
conda create --yes --name TensorFlow2.2 python=3.8
conda activate TensorFlow2.2
conda install --yes cudatoolkit=10.1 cudnn=7.6 -c nvidia
conda install --yes cython matplotlib numpy=1.18 pandas pydot scikit-learn
pip install tensorflow==2.2.3
pip install tf2cv
pip install opencv-python
pip install albumentations --no-binary imgaug,albumentations
```

## Training

```bash
python3 -u solution.py --dataset_name "Market1501" --use_horizontal_flipping_inside_model --nouse_horizontal_flipping_in_evaluation --steps_per_epoch 200 --epoch_num 200
```

- To train on other datasets, replace `"Market1501"` with `"DukeMTMC_reID"` or `"MSMT17"`.
- Specify the arguments for the backbone model:
```bash
--backbone_model_name "resnet50" --learning_rate_start 2e-4 --learning_rate_end 2e-4 --learning_rate_base 2e-4 --learning_rate_lower_bound 2e-6 --kernel_regularization_factor 0.0005 --bias_regularization_factor 0.0005 --gamma_regularization_factor 0.0005 --beta_regularization_factor 0.0005
```
```bash
--backbone_model_name "ibn_resnet50" --learning_rate_start 3e-4 --learning_rate_end 3e-4 --learning_rate_base 3e-4 --learning_rate_lower_bound 3e-6 --kernel_regularization_factor 0.0005 --bias_regularization_factor 0.0005 --gamma_regularization_factor 0.0005 --beta_regularization_factor 0.0005
```
```bash
--backbone_model_name "resnesta50" --learning_rate_start 3e-4 --learning_rate_end 3e-4 --learning_rate_base 3e-4 --learning_rate_lower_bound 3e-6 --kernel_regularization_factor 0.0010 --bias_regularization_factor 0.0010 --gamma_regularization_factor 0.0010 --beta_regularization_factor 0.0010
```
- To evaluate on a subset of the complete test set, append `--testing_size 0.5` to the command. Alternatively, you may turn this feature off by using `--testing_size 0.0`.

## Evaluation

```bash
python3 -u solution.py --dataset_name "Market1501" --backbone_model_name "resnesta50" --pretrained_model_file_path "?.h5" --use_horizontal_flipping_inside_model --use_horizontal_flipping_in_evaluation --output_folder_path "evaluation_only" --evaluation_only --freeze_backbone_for_N_epochs 0 --testing_size 1.0 --evaluate_testing_every_N_epochs 1
```

- Fill in the `pretrained_model_file_path` argument using the h5 file obtained during training.
- To use the re-ranking method, append `--use_re_ranking` to the command.

## Model Zoo

| Dataset | Backbone | mAP | Weights |
| - | - | - |- |
| Market1501 | ResNeSt50 | 89.6 | [Link](https://tuni-my.sharepoint.com/:u:/g/personal/xingyang_ni_tuni_fi/EQo_hFaK_2xBiOkfiiTtePoBvdpO0Fkld-n5EnIgTvtfuw?e=HuM1P4) |
| DukeMTMC_reID | ResNeSt50 | 81.5 | [Link](https://tuni-my.sharepoint.com/:u:/g/personal/xingyang_ni_tuni_fi/EUkbU8F-fMpIsL5Gm9Ou_6YBfgZ-YYJHM2omPeGp8iTIRA?e=MVTzLN) |
| MSMT17 | ResNeSt50 | 68.0 | [Link](https://tuni-my.sharepoint.com/:u:/g/personal/xingyang_ni_tuni_fi/ERTVCkvo8P9OkJc76ae4QN8Bm1Iicu_FELlfG1r-7R0a5g?e=DkRBVr) |

## Acknowledgements

- Evaluation Metrics are adapted from [deep-person-reid](https://github.com/KaiyangZhou/deep-person-reid/blob/v1.0.6/torchreid/metrics/rank_cylib/rank_cy.pyx).
- Re-Ranking is adapted from [person-re-ranking](https://github.com/zhunzhong07/person-re-ranking/blob/master/python-version/re_ranking_ranklist.py).
- Random Grayscale Patch Replacement is adapted from [Data-Augmentation](https://github.com/finger-monkey/Data-Augmentation/blob/main/trans_gray.py).
- Random Erasing is adapted from [Random-Erasing](https://github.com/zhunzhong07/Random-Erasing/blob/master/transforms.py).
- Triplet Loss is adapted from [triplet-reid](https://github.com/VisualComputingInstitute/triplet-reid/blob/master/loss.py).

## Citation

Please consider citing [this work](https://ieeexplore.ieee.org/document/9484010) if it helps your research.

```
@inproceedings{ni2021flipreid,
  author={Ni, Xingyang and Rahtu, Esa},
  booktitle={2021 9th European Workshop on Visual Information Processing (EUVIP)},
  title={FlipReID: Closing the Gap Between Training and Inference in Person Re-Identification},
  year={2021},
  volume={},
  number={},
  pages={1-6},
  doi={10.1109/EUVIP50544.2021.9484010}
}
```