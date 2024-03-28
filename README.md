# Mind The Edge: Refining Depth Edges in Sparsely-Supervised Monocular Depth Estimation
This is the official implementation of the CVPR24' paper "Mind The Edge: Refining Depth Edges in Sparsely-Supervised Monocular Depth
Estimation" by Lior Talker, Aviad Cohen, Erez Yosef, Alexandra Dana and Michael Dinerstein (Samsung Israel Research Center - SIRC).
<h3 align="center"><a href="https://arxiv.org/pdf/2212.05315.pdf">Paper (ArXiv)</a> | Paper (CVPR24') | Supp (CVPR24') | Poster (CVPR24')</h3>

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

## Acknowledgements
<a href="https://github.com/Britefury/py-bsds500">py-bsds500</a>
