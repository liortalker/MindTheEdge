
import os

# f = open('/algo/ws/liort/mindtheedge_refiningdepthedges/data/kitti_de/kitti_115_annotated_edges.txt', 'r')
f = open('/algo/ws/liort/mindtheedge_refiningdepthedges/data/ddad_de/ddad_val_edges_annotated_edges.txt', 'r')
lines = f.readlines()
f.close()

# f = open('/algo/ws/liort/mindtheedge_refiningdepthedges/data/kitti_de/kitti_de_annotated_edges.txt', 'w')
f = open('/algo/ws/liort/mindtheedge_refiningdepthedges/data/ddad_de/ddad_de_annotated_edges.txt', 'w')
lines_new = [x.split('/')[-1] for x in lines]
f.writelines(lines_new)
f.close()

print('Done')
