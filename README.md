# Mind The Edge: Refining Depth Edges in Sparsely-Supervised Monocular Depth Estimation
This is the official implementation of the CVPR24' paper "Mind The Edge: Refining Depth Edges in Sparsely-Supervised Monocular Depth
Estimation" by Lior Talker, Aviad Cohen, Erez Yosef, Alexandra Dana and Michael Dinerstein (Samsung Israel Research Center - SIRC).
<h3 align="center"><a href="https://arxiv.org/pdf/2212.05315.pdf">Paper (ArXiv)</a> | <a href="https://openaccess.thecvf.com/content/CVPR2024/papers/Talker_Mind_The_Edge_Refining_Depth_Edges_in_Sparsely-Supervised_Monocular_Depth_CVPR_2024_paper.pdf"> Paper (CVPR24') | <a href="https://openaccess.thecvf.com/content/CVPR2024/supplemental/Talker_Mind_The_Edge_CVPR_2024_supplemental.pdf"> Supp (CVPR24') | <a href="https://cvpr.thecvf.com/media/PosterPDFs/CVPR%202024/29549.png?t=1715948618.3559048"> Poster (CVPR24') | <a href="https://www.youtube.com/watch?v=88X5mnp3AMo&t=242s">Video (CVPR24')</a></h3>


<br />

> **Abstract:** *Monocular Depth Estimation (MDE) is a fundamental problem in computer vision with numerous applications. Recently, LIDAR-supervised methods have achieved remarkable per-pixel depth accuracy in outdoor scenes. However, significant errors are typically found in the proximity of depth discontinuities, i.e., depth edges, which often hinder the performance of depth-dependent applications that are sensitive to such inaccuracies, e.g., novel view synthesis and augmented reality. Since direct supervision for the location of depth edges is typically unavailable in sparse LIDAR-based scenes, encouraging the MDE model to produce correct depth edges is not straightforward. To the best of our knowledge this paper is the first attempt to address the depth edges issue for LIDAR-supervised scenes. In this work we propose to learn to detect the location of depth edges from densely-supervised synthetic data, and use it to generate supervision for the depth edges in the MDE training. %Despite the ’domain gap’ between synthetic and real data, we show that depth edges that are estimated directly are significantly more accurate than the ones that emerge indirectly from the MDE training. To quantitatively evaluate our approach, and due to the lack of depth edges ground truth in LIDAR-based scenes, we manually annotated subsets of the KITTI and the DDAD datasets with depth edges ground truth. We demonstrate significant gains in the accuracy of the depth edges with comparable per-pixel depth accuracy on several challenging datasets.* 

## Inference
To run Packnet-SAN, trained on KITTI, with our edge loss, on KITTI Depth Edges (KITTI-DE):
1. Download the [trained weights](https://drive.google.com/file/d/1gSM-sE4-ssW_Syz4fZs81hp89SnFmqFd/view?usp=sharing) (or train your own weights - see Training setion),  and place in ./checkpoints folder.
2. Run inference: 

    ```bash
    python infer_edges.py --config 'packnet_code/configs/infer_packnet_kitti.yaml'
    ```
    (Edit ./packnet_code/configs/infer_packnet_kitti.yaml to change output path, etc)
3. The results are generated (by default) in ./results

## Training
To train Packnet-SAN with our edge loss on KITTI:
1. Download KITTI's RGB and depth images () and place in ./data/kitti/rgb and ./data/kitti/depth, respectively.
2. Resize the RGB and depth to 384x1280, using e.g., cv2.resize and [`resize_depth_preserve`](http://gitlab-srv/red-team/MindTheEdge/-/blob/main/packnet_code/packnet_sfm/datasets/augmentations.py#L56), respectively.
3. Download the [DEE network (trained on GTA)](https://drive.google.com/file/d/17BbJqfKjrYqjWw6SK5nbidGOLemdpYYE/view?usp=sharing) for annotating the depth edges of KITTI's training set, and place in ./checkpoints.
4. Run depth edge annotation to produce the depth edges and normals:

    ```bash
    python infer_edge_estimation.py --config 'packnet_code/configs/annotate_edges_kitti_training_set.yaml'
    ```
    (Edit ./packnet_code/configs/annotate_edges_kitti_training_set.yaml to change output path, etc)
5. The results are generated (by default) in ./results/DEE_estimated_depth_edges_kitti_train, and a split file (./results/DEE_estimated_depth_edges_kitti_train/rgb_lidar_edges_split.txt) will be created for training (step 6).
6. Download [Packnet-SAN supervised pretraining](https://tri-ml-public.s3.amazonaws.com/github/packnet-sfm/models/PackNetSAN01_HR_sup_K.ckpt) and place in ./checkpoints.
7. Run training:
     ```bash
    python train_edges.py 'packnet_code/configs/train_packnet_san_kitti_with_edges.yaml'
    ```
    (Edit ./packnet_code/configs/train_packnet_san_kitti_with_edges.yaml to change output path, etc)

## Datasets

The KITTI Depth Edges (KITTI-DE) and DDAD Depth Edges (DDAD-DE) validation datasets from the paper are provided as binary edge images in data/kitti_de/gt and data/ddad_de/gt.
Text lists with a GT image per-line are stored in data/kitti_de/kitti_de_annotated_edges.txt and data/ddad_de/ddad_de_annotated_edges.txt.

The RGB images for KITTI-DE are taken from the <a href="https://www.cvlibs.net/datasets/kitti/eval_instance_seg.php?benchmark=instanceSeg2015">KITTI's Semantic Instance Segmentation Evaluation benchmark</a>.
The corresponding RGB images have the same filename as the filenames in the KITTI-DE validation set.

The RGB images for DDAD-DE are taken from the <a href="https://tri-ml-public.s3.amazonaws.com/github/DDAD/datasets/DDAD.tar">DDAD's instance segmentation images in the validation set</a>.
(The validation set corresponds to clips 150-199, where in each clip, one image has instance segmentation GT, from which we annotated the depth edge GT.)

To evaluate depth maps (in .npy format) on the KITTI-DE dataset using the AUC (edges) metric (as in Tab.1 in the paper):
```bash
python eval_depth_edges.py 
--depth_pred_list_path [path_to_pred_npy_name_list] 
--depth_pred_dir_path [path_to_dir_with_pred_npy_files]
--depth_edge_gt_list_path data/kitti_de/kitti_de_annotated_edges.txt
--depth_edge_gt_dir_path data/kitti_de/gt
```
- *[path_to_pred_npy_name_list]* is a path to a txt file with the *names only* of the predicted depth .npy files.
- *[path_to_dir_with_pred_npy_files]* is a path to the directory that contains the predicted depth .npy files.
- Note: the order of the predicted depth and the depth edge GT must be the same.

To evaluate depth maps (in .npy format) on the DDAD-DE dataset using the AUC (edges) metric:
1. Please download and compile the [py-bsds500 evaluation suite](https://github.com/Britefury/py-bsds500)
2. Place the source and compiled files under /bsds_metric.
3. Run: 
    ```bash
    python eval_depth_edges.py  
    --depth_pred_list_path [path_to_pred_npy_name_list] 
    --depth_pred_dir_path [path_to_dir_with_pred_npy_files]
    --depth_edge_gt_list_path data/ddad_de/ddad_de_annotated_edges.txt
    --depth_edge_gt_dir_path data/ddad_de/gt
    --prec_recall_eval_range_min
    0.14
    --prec_recall_eval_range_max
    0.37
    ```
- *prec_recall_eval_range_min* and *prec_recall_eval_range_max* are the (partial) range in which the edge AUC metric is computed (as in Tab.2 in the paper).

## License
Copyright (c) 2024 Samsung Israel Research Center (SIRC).

The part of the code that implements the "Mind The Edge" CVPR24' paper is provided under the "Creative Commons Attribution-NonCommercial-ShareAlike 4.0 International" (CC BY-NC-SA 4.0) license (see <http://creativecommons.org/licenses/by-nc-sa/4.0/>).
**Please check the first line of each file for its specific license.**

Another part of the code is based on Packnet-SAN (https://github.com/TRI-ML/packnet-sfm), which is provided under the MIT license:

Copyright (c) 2019 Toyota Research Institute (TRI)

Permission is hereby granted, free of charge, to any person obtaining a copy of this software and associated documentation files (the "Software"), to deal in the Software without restriction, including without limitation the rights to use, copy, modify, merge, publish, distribute, sublicense, and/or sell copies of the Software, and to permit persons to whom the Software is furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in all copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY, FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM, OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE SOFTWARE.



## Citation
If you find this work relevant, please consider citing:

    @inproceedings{talker2024mind,
      title={Mind The Edge: Refining Depth Edges in Sparsely-Supervised Monocular Depth Estimation},
      author={Talker, Lior and Cohen, Aviad and Yosef, Erez and Dana, Alexandra and Dinerstein, Michael},
      booktitle={Proceedings of the IEEE/CVF conference on computer vision and pattern recognition},
      year={2024}
    }

## Acknowledgements
<a href="https://github.com/TRI-ML/packnet-sfm">Packnet-SAN</a>

<a href="https://github.com/Britefury/py-bsds500">py-bsds500</a>