import os
import numpy as np
from trackit.datasets.SOT.constructor import SingleObjectTrackingDatasetConstructor
from trackit.datasets.common.utils.dtype import try_get_int_array


_uav123_sequence_info_list = [
    {"name": "bike1", "path": "data_seq/UAV123/bike1", "startFrame": 1, "endFrame": 3085, "nz": 6,
     "ext": "jpg", "anno_path": "anno/UAV123/bike1.txt", "fps": 30, "object_class": "vehicle"},
    {"name": "bike2", "path": "data_seq/UAV123/bike2", "startFrame": 1, "endFrame": 553, "nz": 6,
     "ext": "jpg", "anno_path": "anno/UAV123/bike2.txt", "fps": 30, "object_class": "vehicle"},
    {"name": "bike3", "path": "data_seq/UAV123/bike3", "startFrame": 1, "endFrame": 433, "nz": 6,
     "ext": "jpg", "anno_path": "anno/UAV123/bike3.txt", "fps": 30, "object_class": "vehicle"},
    {"name": "bird1_1", "path": "data_seq/UAV123/bird1", "startFrame": 1, "endFrame": 253, "nz": 6,
     "ext": "jpg", "anno_path": "anno/UAV123/bird1_1.txt", "fps": 30, "object_class": "bird"},
    {"name": "bird1_2", "path": "data_seq/UAV123/bird1", "startFrame": 775, "endFrame": 1477, "nz": 6,
     "ext": "jpg", "anno_path": "anno/UAV123/bird1_2.txt", "fps": 30, "object_class": "bird"},
    {"name": "bird1_3", "path": "data_seq/UAV123/bird1", "startFrame": 1573, "endFrame": 2437, "nz": 6,
     "ext": "jpg", "anno_path": "anno/UAV123/bird1_3.txt", "fps": 30, "object_class": "bird"},
    {"name": "boat1", "path": "data_seq/UAV123/boat1", "startFrame": 1, "endFrame": 901, "nz": 6,
     "ext": "jpg", "anno_path": "anno/UAV123/boat1.txt", "fps": 30, "object_class": "vessel"},
    {"name": "boat2", "path": "data_seq/UAV123/boat2", "startFrame": 1, "endFrame": 799, "nz": 6,
     "ext": "jpg", "anno_path": "anno/UAV123/boat2.txt", "fps": 30, "object_class": "vessel"},
    {"name": "boat3", "path": "data_seq/UAV123/boat3", "startFrame": 1, "endFrame": 901, "nz": 6,
     "ext": "jpg", "anno_path": "anno/UAV123/boat3.txt", "fps": 30, "object_class": "vessel"},
    {"name": "boat4", "path": "data_seq/UAV123/boat4", "startFrame": 1, "endFrame": 553, "nz": 6,
     "ext": "jpg", "anno_path": "anno/UAV123/boat4.txt", "fps": 30, "object_class": "vessel"},
    {"name": "boat5", "path": "data_seq/UAV123/boat5", "startFrame": 1, "endFrame": 505, "nz": 6,
     "ext": "jpg", "anno_path": "anno/UAV123/boat5.txt", "fps": 30, "object_class": "vessel"},
    {"name": "boat6", "path": "data_seq/UAV123/boat6", "startFrame": 1, "endFrame": 805, "nz": 6,
     "ext": "jpg", "anno_path": "anno/UAV123/boat6.txt", "fps": 30, "object_class": "vessel"},
    {"name": "boat7", "path": "data_seq/UAV123/boat7", "startFrame": 1, "endFrame": 535, "nz": 6,
     "ext": "jpg", "anno_path": "anno/UAV123/boat7.txt", "fps": 96, "object_class": "vessel"},
    {"name": "boat8", "path": "data_seq/UAV123/boat8", "startFrame": 1, "endFrame": 685, "nz": 6,
     "ext": "jpg", "anno_path": "anno/UAV123/boat8.txt", "fps": 96, "object_class": "vessel"},
    {"name": "boat9", "path": "data_seq/UAV123/boat9", "startFrame": 1, "endFrame": 1399, "nz": 6,
     "ext": "jpg", "anno_path": "anno/UAV123/boat9.txt", "fps": 96, "object_class": "vessel"},
    {"name": "building1", "path": "data_seq/UAV123/building1", "startFrame": 1, "endFrame": 469, "nz": 6,
     "ext": "jpg", "anno_path": "anno/UAV123/building1.txt", "fps": 30, "object_class": "other"},
    {"name": "building2", "path": "data_seq/UAV123/building2", "startFrame": 1, "endFrame": 577, "nz": 6,
     "ext": "jpg", "anno_path": "anno/UAV123/building2.txt", "fps": 30, "object_class": "other"},
    {"name": "building3", "path": "data_seq/UAV123/building3", "startFrame": 1, "endFrame": 829, "nz": 6,
     "ext": "jpg", "anno_path": "anno/UAV123/building3.txt", "fps": 30, "object_class": "other"},
    {"name": "building4", "path": "data_seq/UAV123/building4", "startFrame": 1, "endFrame": 787, "nz": 6,
     "ext": "jpg", "anno_path": "anno/UAV123/building4.txt", "fps": 30, "object_class": "other"},
    {"name": "building5", "path": "data_seq/UAV123/building5", "startFrame": 1, "endFrame": 481, "nz": 6,
     "ext": "jpg", "anno_path": "anno/UAV123/building5.txt", "fps": 96, "object_class": "other"},
    {"name": "car1_1", "path": "data_seq/UAV123/car1", "startFrame": 1, "endFrame": 751, "nz": 6,
     "ext": "jpg", "anno_path": "anno/UAV123/car1_1.txt", "fps": 96, "object_class": "car"},
    {"name": "car1_2", "path": "data_seq/UAV123/car1", "startFrame": 751, "endFrame": 1627, "nz": 6,
     "ext": "jpg", "anno_path": "anno/UAV123/car1_2.txt", "fps": 96, "object_class": "car"},
    {"name": "car1_3", "path": "data_seq/UAV123/car1", "startFrame": 1627, "endFrame": 2629, "nz": 6,
     "ext": "jpg", "anno_path": "anno/UAV123/car1_3.txt", "fps": 96, "object_class": "car"},
    {"name": "car2", "path": "data_seq/UAV123/car2", "startFrame": 1, "endFrame": 1321, "nz": 6,
     "ext": "jpg", "anno_path": "anno/UAV123/car2.txt", "fps": 96, "object_class": "car"},
    {"name": "car3", "path": "data_seq/UAV123/car3", "startFrame": 1, "endFrame": 1717, "nz": 6,
     "ext": "jpg", "anno_path": "anno/UAV123/car3.txt", "fps": 96, "object_class": "car"},
    {"name": "car4", "path": "data_seq/UAV123/car4", "startFrame": 1, "endFrame": 1345, "nz": 6,
     "ext": "jpg", "anno_path": "anno/UAV123/car4.txt", "fps": 96, "object_class": "car"},
    {"name": "car5", "path": "data_seq/UAV123/car5", "startFrame": 1, "endFrame": 745, "nz": 6,
     "ext": "jpg", "anno_path": "anno/UAV123/car5.txt", "fps": 96, "object_class": "car"},
    {"name": "car6_1", "path": "data_seq/UAV123/car6", "startFrame": 1, "endFrame": 487, "nz": 6,
     "ext": "jpg", "anno_path": "anno/UAV123/car6_1.txt", "fps": 96, "object_class": "car"},
    {"name": "car6_2", "path": "data_seq/UAV123/car6", "startFrame": 487, "endFrame": 1807, "nz": 6,
     "ext": "jpg", "anno_path": "anno/UAV123/car6_2.txt", "fps": 96, "object_class": "car"},
    {"name": "car6_3", "path": "data_seq/UAV123/car6", "startFrame": 1807, "endFrame": 2953, "nz": 6,
     "ext": "jpg", "anno_path": "anno/UAV123/car6_3.txt", "fps": 96, "object_class": "car"},
    {"name": "car6_4", "path": "data_seq/UAV123/car6", "startFrame": 2953, "endFrame": 3925, "nz": 6,
     "ext": "jpg", "anno_path": "anno/UAV123/car6_4.txt", "fps": 96, "object_class": "car"},
    {"name": "car6_5", "path": "data_seq/UAV123/car6", "startFrame": 3925, "endFrame": 4861, "nz": 6,
     "ext": "jpg", "anno_path": "anno/UAV123/car6_5.txt", "fps": 96, "object_class": "car"},
    {"name": "car7", "path": "data_seq/UAV123/car7", "startFrame": 1, "endFrame": 1033, "nz": 6,
     "ext": "jpg", "anno_path": "anno/UAV123/car7.txt", "fps": 96, "object_class": "car"},
    {"name": "car8_1", "path": "data_seq/UAV123/car8", "startFrame": 1, "endFrame": 1357, "nz": 6,
     "ext": "jpg", "anno_path": "anno/UAV123/car8_1.txt", "fps": 96, "object_class": "car"},
    {"name": "car8_2", "path": "data_seq/UAV123/car8", "startFrame": 1357, "endFrame": 2575, "nz": 6,
     "ext": "jpg", "anno_path": "anno/UAV123/car8_2.txt", "fps": 96, "object_class": "car"},
    {"name": "car9", "path": "data_seq/UAV123/car9", "startFrame": 1, "endFrame": 1879, "nz": 6,
     "ext": "jpg", "anno_path": "anno/UAV123/car9.txt", "fps": 96, "object_class": "car"},
    {"name": "car10", "path": "data_seq/UAV123/car10", "startFrame": 1, "endFrame": 1405, "nz": 6,
     "ext": "jpg", "anno_path": "anno/UAV123/car10.txt", "fps": 96, "object_class": "car"},
    {"name": "car11", "path": "data_seq/UAV123/car11", "startFrame": 1, "endFrame": 337, "nz": 6,
     "ext": "jpg", "anno_path": "anno/UAV123/car11.txt", "fps": 60, "object_class": "car"},
    {"name": "car12", "path": "data_seq/UAV123/car12", "startFrame": 1, "endFrame": 499, "nz": 6,
     "ext": "jpg", "anno_path": "anno/UAV123/car12.txt", "fps": 96, "object_class": "car"},
    {"name": "car13", "path": "data_seq/UAV123/car13", "startFrame": 1, "endFrame": 415, "nz": 6,
     "ext": "jpg", "anno_path": "anno/UAV123/car13.txt", "fps": 96, "object_class": "car"},
    {"name": "car14", "path": "data_seq/UAV123/car14", "startFrame": 1, "endFrame": 1327, "nz": 6,
     "ext": "jpg", "anno_path": "anno/UAV123/car14.txt", "fps": 96, "object_class": "car"},
    {"name": "car15", "path": "data_seq/UAV123/car15", "startFrame": 1, "endFrame": 469, "nz": 6,
     "ext": "jpg", "anno_path": "anno/UAV123/car15.txt", "fps": 30, "object_class": "car"},
    {"name": "car16_1", "path": "data_seq/UAV123/car16", "startFrame": 1, "endFrame": 415, "nz": 6,
     "ext": "jpg", "anno_path": "anno/UAV123/car16_1.txt", "fps": 96, "object_class": "car"},
    {"name": "car16_2", "path": "data_seq/UAV123/car16", "startFrame": 415, "endFrame": 1993, "nz": 6,
     "ext": "jpg", "anno_path": "anno/UAV123/car16_2.txt", "fps": 96, "object_class": "car"},
    {"name": "car17", "path": "data_seq/UAV123/car17", "startFrame": 1, "endFrame": 1057, "nz": 6,
     "ext": "jpg", "anno_path": "anno/UAV123/car17.txt", "fps": 96, "object_class": "car"},
    {"name": "car18", "path": "data_seq/UAV123/car18", "startFrame": 1, "endFrame": 1207, "nz": 6,
     "ext": "jpg", "anno_path": "anno/UAV123/car18.txt", "fps": 96, "object_class": "car"},
    {"name": "group1_1", "path": "data_seq/UAV123/group1", "startFrame": 1, "endFrame": 1333, "nz": 6,
     "ext": "jpg", "anno_path": "anno/UAV123/group1_1.txt", "fps": 30, "object_class": "person"},
    {"name": "group1_2", "path": "data_seq/UAV123/group1", "startFrame": 1333, "endFrame": 2515, "nz": 6,
     "ext": "jpg", "anno_path": "anno/UAV123/group1_2.txt", "fps": 30, "object_class": "person"},
    {"name": "group1_3", "path": "data_seq/UAV123/group1", "startFrame": 2515, "endFrame": 3925, "nz": 6,
     "ext": "jpg", "anno_path": "anno/UAV123/group1_3.txt", "fps": 30, "object_class": "person"},
    {"name": "group1_4", "path": "data_seq/UAV123/group1", "startFrame": 3925, "endFrame": 4873, "nz": 6,
     "ext": "jpg", "anno_path": "anno/UAV123/group1_4.txt", "fps": 30, "object_class": "person"},
    {"name": "group2_1", "path": "data_seq/UAV123/group2", "startFrame": 1, "endFrame": 907, "nz": 6,
     "ext": "jpg", "anno_path": "anno/UAV123/group2_1.txt", "fps": 96, "object_class": "person"},
    {"name": "group2_2", "path": "data_seq/UAV123/group2", "startFrame": 907, "endFrame": 1771, "nz": 6,
     "ext": "jpg", "anno_path": "anno/UAV123/group2_2.txt", "fps": 96, "object_class": "person"},
    {"name": "group2_3", "path": "data_seq/UAV123/group2", "startFrame": 1771, "endFrame": 2683, "nz": 6,
     "ext": "jpg", "anno_path": "anno/UAV123/group2_3.txt", "fps": 96, "object_class": "person"},
    {"name": "group3_1", "path": "data_seq/UAV123/group3", "startFrame": 1, "endFrame": 1567, "nz": 6,
     "ext": "jpg", "anno_path": "anno/UAV123/group3_1.txt", "fps": 96, "object_class": "person"},
    {"name": "group3_2", "path": "data_seq/UAV123/group3", "startFrame": 1567, "endFrame": 2827, "nz": 6,
     "ext": "jpg", "anno_path": "anno/UAV123/group3_2.txt", "fps": 96, "object_class": "person"},
    {"name": "group3_3", "path": "data_seq/UAV123/group3", "startFrame": 2827, "endFrame": 4369, "nz": 6,
     "ext": "jpg", "anno_path": "anno/UAV123/group3_3.txt", "fps": 96, "object_class": "person"},
    {"name": "group3_4", "path": "data_seq/UAV123/group3", "startFrame": 4369, "endFrame": 5527, "nz": 6,
     "ext": "jpg", "anno_path": "anno/UAV123/group3_4.txt", "fps": 96, "object_class": "person"},
    {"name": "person1", "path": "data_seq/UAV123/person1", "startFrame": 1, "endFrame": 799, "nz": 6,
     "ext": "jpg", "anno_path": "anno/UAV123/person1.txt", "fps": 30, "object_class": "person"},
    {"name": "person2_1", "path": "data_seq/UAV123/person2", "startFrame": 1, "endFrame": 1189, "nz": 6,
     "ext": "jpg", "anno_path": "anno/UAV123/person2_1.txt", "fps": 30, "object_class": "person"},
    {"name": "person2_2", "path": "data_seq/UAV123/person2", "startFrame": 1189, "endFrame": 2623, "nz": 6,
     "ext": "jpg", "anno_path": "anno/UAV123/person2_2.txt", "fps": 30, "object_class": "person"},
    {"name": "person3", "path": "data_seq/UAV123/person3", "startFrame": 1, "endFrame": 643, "nz": 6,
     "ext": "jpg", "anno_path": "anno/UAV123/person3.txt", "fps": 30, "object_class": "person"},
    {"name": "person4_1", "path": "data_seq/UAV123/person4", "startFrame": 1, "endFrame": 1501, "nz": 6,
     "ext": "jpg", "anno_path": "anno/UAV123/person4_1.txt", "fps": 30, "object_class": "person"},
    {"name": "person4_2", "path": "data_seq/UAV123/person4", "startFrame": 1501, "endFrame": 2743, "nz": 6,
     "ext": "jpg", "anno_path": "anno/UAV123/person4_2.txt", "fps": 30, "object_class": "person"},
    {"name": "person5_1", "path": "data_seq/UAV123/person5", "startFrame": 1, "endFrame": 877, "nz": 6,
     "ext": "jpg", "anno_path": "anno/UAV123/person5_1.txt", "fps": 30, "object_class": "person"},
    {"name": "person5_2", "path": "data_seq/UAV123/person5", "startFrame": 877, "endFrame": 2101, "nz": 6,
     "ext": "jpg", "anno_path": "anno/UAV123/person5_2.txt", "fps": 30, "object_class": "person"},
    {"name": "person6", "path": "data_seq/UAV123/person6", "startFrame": 1, "endFrame": 901, "nz": 6,
     "ext": "jpg", "anno_path": "anno/UAV123/person6.txt", "fps": 30, "object_class": "person"},
    {"name": "person7_1", "path": "data_seq/UAV123/person7", "startFrame": 1, "endFrame": 1249, "nz": 6,
     "ext": "jpg", "anno_path": "anno/UAV123/person7_1.txt", "fps": 30, "object_class": "person"},
    {"name": "person7_2", "path": "data_seq/UAV123/person7", "startFrame": 1249, "endFrame": 2065, "nz": 6,
     "ext": "jpg", "anno_path": "anno/UAV123/person7_2.txt", "fps": 30, "object_class": "person"},
    {"name": "person8_1", "path": "data_seq/UAV123/person8", "startFrame": 1, "endFrame": 1075, "nz": 6,
     "ext": "jpg", "anno_path": "anno/UAV123/person8_1.txt", "fps": 30, "object_class": "person"},
    {"name": "person8_2", "path": "data_seq/UAV123/person8", "startFrame": 1075, "endFrame": 1525, "nz": 6,
     "ext": "jpg", "anno_path": "anno/UAV123/person8_2.txt", "fps": 30, "object_class": "person"},
    {"name": "person9", "path": "data_seq/UAV123/person9", "startFrame": 1, "endFrame": 661, "nz": 6,
     "ext": "jpg", "anno_path": "anno/UAV123/person9.txt", "fps": 30, "object_class": "person"},
    {"name": "person10", "path": "data_seq/UAV123/person10", "startFrame": 1, "endFrame": 1021, "nz": 6,
     "ext": "jpg", "anno_path": "anno/UAV123/person10.txt", "fps": 30, "object_class": "person"},
    {"name": "person11", "path": "data_seq/UAV123/person11", "startFrame": 1, "endFrame": 721, "nz": 6,
     "ext": "jpg", "anno_path": "anno/UAV123/person11.txt", "fps": 30, "object_class": "person"},
    {"name": "person12_1", "path": "data_seq/UAV123/person12", "startFrame": 1, "endFrame": 601, "nz": 6,
     "ext": "jpg", "anno_path": "anno/UAV123/person12_1.txt", "fps": 30, "object_class": "person"},
    {"name": "person12_2", "path": "data_seq/UAV123/person12", "startFrame": 601, "endFrame": 1621, "nz": 6,
     "ext": "jpg", "anno_path": "anno/UAV123/person12_2.txt", "fps": 30, "object_class": "person"},
    {"name": "person13", "path": "data_seq/UAV123/person13", "startFrame": 1, "endFrame": 883, "nz": 6,
     "ext": "jpg", "anno_path": "anno/UAV123/person13.txt", "fps": 30, "object_class": "person"},
    {"name": "person14_1", "path": "data_seq/UAV123/person14", "startFrame": 1, "endFrame": 847, "nz": 6,
     "ext": "jpg", "anno_path": "anno/UAV123/person14_1.txt", "fps": 96, "object_class": "person"},
    {"name": "person14_2", "path": "data_seq/UAV123/person14", "startFrame": 847, "endFrame": 1813, "nz": 6,
     "ext": "jpg", "anno_path": "anno/UAV123/person14_2.txt", "fps": 96, "object_class": "person"},
    {"name": "person14_3", "path": "data_seq/UAV123/person14", "startFrame": 1813, "endFrame": 2923,
     "nz": 6, "ext": "jpg", "anno_path": "anno/UAV123/person14_3.txt", "fps": 96, "object_class": "person"},
    {"name": "person15", "path": "data_seq/UAV123/person15", "startFrame": 1, "endFrame": 1339, "nz": 6,
     "ext": "jpg", "anno_path": "anno/UAV123/person15.txt", "fps": 96, "object_class": "person"},
    {"name": "person16", "path": "data_seq/UAV123/person16", "startFrame": 1, "endFrame": 1147, "nz": 6,
     "ext": "jpg", "anno_path": "anno/UAV123/person16.txt", "fps": 96, "object_class": "person"},
    {"name": "person17_1", "path": "data_seq/UAV123/person17", "startFrame": 1, "endFrame": 1501, "nz": 6,
     "ext": "jpg", "anno_path": "anno/UAV123/person17_1.txt", "fps": 96, "object_class": "person"},
    {"name": "person17_2", "path": "data_seq/UAV123/person17", "startFrame": 1501, "endFrame": 2347,
     "nz": 6, "ext": "jpg", "anno_path": "anno/UAV123/person17_2.txt", "fps": 96, "object_class": "person"},
    {"name": "person18", "path": "data_seq/UAV123/person18", "startFrame": 1, "endFrame": 1393, "nz": 6,
     "ext": "jpg", "anno_path": "anno/UAV123/person18.txt", "fps": 96, "object_class": "person"},
    {"name": "person19_1", "path": "data_seq/UAV123/person19", "startFrame": 1, "endFrame": 1243, "nz": 6,
     "ext": "jpg", "anno_path": "anno/UAV123/person19_1.txt", "fps": 96, "object_class": "person"},
    {"name": "person19_2", "path": "data_seq/UAV123/person19", "startFrame": 1243, "endFrame": 2791,
     "nz": 6, "ext": "jpg", "anno_path": "anno/UAV123/person19_2.txt", "fps": 96, "object_class": "person"},
    {"name": "person19_3", "path": "data_seq/UAV123/person19", "startFrame": 2791, "endFrame": 4357,
     "nz": 6, "ext": "jpg", "anno_path": "anno/UAV123/person19_3.txt", "fps": 96, "object_class": "person"},
    {"name": "person20", "path": "data_seq/UAV123/person20", "startFrame": 1, "endFrame": 1783, "nz": 6,
     "ext": "jpg", "anno_path": "anno/UAV123/person20.txt", "fps": 96, "object_class": "person"},
    {"name": "person21", "path": "data_seq/UAV123/person21", "startFrame": 1, "endFrame": 487, "nz": 6,
     "ext": "jpg", "anno_path": "anno/UAV123/person21.txt", "fps": 30, "object_class": "person"},
    {"name": "person22", "path": "data_seq/UAV123/person22", "startFrame": 1, "endFrame": 199, "nz": 6,
     "ext": "jpg", "anno_path": "anno/UAV123/person22.txt", "fps": 30, "object_class": "person"},
    {"name": "person23", "path": "data_seq/UAV123/person23", "startFrame": 1, "endFrame": 397, "nz": 6,
     "ext": "jpg", "anno_path": "anno/UAV123/person23.txt", "fps": 30, "object_class": "person"},
    {"name": "truck1", "path": "data_seq/UAV123/truck1", "startFrame": 1, "endFrame": 463, "nz": 6,
     "ext": "jpg", "anno_path": "anno/UAV123/truck1.txt", "fps": 30, "object_class": "truck"},
    {"name": "truck2", "path": "data_seq/UAV123/truck2", "startFrame": 1, "endFrame": 385, "nz": 6,
     "ext": "jpg", "anno_path": "anno/UAV123/truck2.txt", "fps": 30, "object_class": "truck"},
    {"name": "truck3", "path": "data_seq/UAV123/truck3", "startFrame": 1, "endFrame": 535, "nz": 6,
     "ext": "jpg", "anno_path": "anno/UAV123/truck3.txt", "fps": 30, "object_class": "truck"},
    {"name": "truck4_1", "path": "data_seq/UAV123/truck4", "startFrame": 1, "endFrame": 577, "nz": 6,
     "ext": "jpg", "anno_path": "anno/UAV123/truck4_1.txt", "fps": 30, "object_class": "truck"},
    {"name": "truck4_2", "path": "data_seq/UAV123/truck4", "startFrame": 577, "endFrame": 1261, "nz": 6,
     "ext": "jpg", "anno_path": "anno/UAV123/truck4_2.txt", "fps": 30, "object_class": "truck"},
    {"name": "uav1_1", "path": "data_seq/UAV123/uav1", "startFrame": 1, "endFrame": 1555, "nz": 6,
     "ext": "jpg", "anno_path": "anno/UAV123/uav1_1.txt", "fps": 30, "object_class": "aircraft"},
    {"name": "uav1_2", "path": "data_seq/UAV123/uav1", "startFrame": 1555, "endFrame": 2377, "nz": 6,
     "ext": "jpg", "anno_path": "anno/UAV123/uav1_2.txt", "fps": 30, "object_class": "aircraft"},
    {"name": "uav1_3", "path": "data_seq/UAV123/uav1", "startFrame": 2473, "endFrame": 3469, "nz": 6,
     "ext": "jpg", "anno_path": "anno/UAV123/uav1_3.txt", "fps": 30, "object_class": "aircraft"},
    {"name": "uav2", "path": "data_seq/UAV123/uav2", "startFrame": 1, "endFrame": 133, "nz": 6,
     "ext": "jpg", "anno_path": "anno/UAV123/uav2.txt", "fps": 30, "object_class": "aircraft"},
    {"name": "uav3", "path": "data_seq/UAV123/uav3", "startFrame": 1, "endFrame": 265, "nz": 6,
     "ext": "jpg", "anno_path": "anno/UAV123/uav3.txt", "fps": 30, "object_class": "aircraft"},
    {"name": "uav4", "path": "data_seq/UAV123/uav4", "startFrame": 1, "endFrame": 157, "nz": 6,
     "ext": "jpg", "anno_path": "anno/UAV123/uav4.txt", "fps": 30, "object_class": "aircraft"},
    {"name": "uav5", "path": "data_seq/UAV123/uav5", "startFrame": 1, "endFrame": 139, "nz": 6,
     "ext": "jpg", "anno_path": "anno/UAV123/uav5.txt", "fps": 30, "object_class": "aircraft"},
    {"name": "uav6", "path": "data_seq/UAV123/uav6", "startFrame": 1, "endFrame": 109, "nz": 6,
     "ext": "jpg", "anno_path": "anno/UAV123/uav6.txt", "fps": 30, "object_class": "aircraft"},
    {"name": "uav7", "path": "data_seq/UAV123/uav7", "startFrame": 1, "endFrame": 373, "nz": 6,
     "ext": "jpg", "anno_path": "anno/UAV123/uav7.txt", "fps": 30, "object_class": "aircraft"},
    {"name": "uav8", "path": "data_seq/UAV123/uav8", "startFrame": 1, "endFrame": 301, "nz": 6,
     "ext": "jpg", "anno_path": "anno/UAV123/uav8.txt", "fps": 30, "object_class": "aircraft"},
    {"name": "wakeboard1", "path": "data_seq/UAV123/wakeboard1", "startFrame": 1, "endFrame": 421, "nz": 6,
     "ext": "jpg", "anno_path": "anno/UAV123/wakeboard1.txt", "fps": 96, "object_class": "person"},
    {"name": "wakeboard2", "path": "data_seq/UAV123/wakeboard2", "startFrame": 1, "endFrame": 733, "nz": 6,
     "ext": "jpg", "anno_path": "anno/UAV123/wakeboard2.txt", "fps": 96, "object_class": "person"},
    {"name": "wakeboard3", "path": "data_seq/UAV123/wakeboard3", "startFrame": 1, "endFrame": 823, "nz": 6,
     "ext": "jpg", "anno_path": "anno/UAV123/wakeboard3.txt", "fps": 96, "object_class": "person"},
    {"name": "wakeboard4", "path": "data_seq/UAV123/wakeboard4", "startFrame": 1, "endFrame": 697, "nz": 6,
     "ext": "jpg", "anno_path": "anno/UAV123/wakeboard4.txt", "fps": 96, "object_class": "person"},
    {"name": "wakeboard5", "path": "data_seq/UAV123/wakeboard5", "startFrame": 1, "endFrame": 1675, "nz": 6,
     "ext": "jpg", "anno_path": "anno/UAV123/wakeboard5.txt", "fps": 96, "object_class": "person"},
    {"name": "wakeboard6", "path": "data_seq/UAV123/wakeboard6", "startFrame": 1, "endFrame": 1165, "nz": 6,
     "ext": "jpg", "anno_path": "anno/UAV123/wakeboard6.txt", "fps": 96, "object_class": "person"},
    {"name": "wakeboard7", "path": "data_seq/UAV123/wakeboard7", "startFrame": 1, "endFrame": 199, "nz": 6,
     "ext": "jpg", "anno_path": "anno/UAV123/wakeboard7.txt", "fps": 96, "object_class": "person"},
    {"name": "wakeboard8", "path": "data_seq/UAV123/wakeboard8", "startFrame": 1, "endFrame": 1543, "nz": 6,
     "ext": "jpg", "anno_path": "anno/UAV123/wakeboard8.txt", "fps": 96, "object_class": "person"},
    {"name": "wakeboard9", "path": "data_seq/UAV123/wakeboard9", "startFrame": 1, "endFrame": 355, "nz": 6,
     "ext": "jpg", "anno_path": "anno/UAV123/wakeboard9.txt", "fps": 96, "object_class": "person"},
    {"name": "wakeboard10", "path": "data_seq/UAV123/wakeboard10", "startFrame": 1, "endFrame": 469,
     "nz": 6, "ext": "jpg", "anno_path": "anno/UAV123/wakeboard10.txt", "fps": 96, "object_class": "person"},
    {"name": "car1_s", "path": "data_seq/UAV123/car1_s", "startFrame": 1, "endFrame": 1475, "nz": 6,
     "ext": "jpg", "anno_path": "anno/UAV123/car1_s.txt", "fps": 30, "object_class": "car"},
    {"name": "car2_s", "path": "data_seq/UAV123/car2_s", "startFrame": 1, "endFrame": 320, "nz": 6,
     "ext": "jpg", "anno_path": "anno/UAV123/car2_s.txt", "fps": 30, "object_class": "car"},
    {"name": "car3_s", "path": "data_seq/UAV123/car3_s", "startFrame": 1, "endFrame": 1300, "nz": 6,
     "ext": "jpg", "anno_path": "anno/UAV123/car3_s.txt", "fps": 30, "object_class": "car"},
    {"name": "car4_s", "path": "data_seq/UAV123/car4_s", "startFrame": 1, "endFrame": 830, "nz": 6,
     "ext": "jpg", "anno_path": "anno/UAV123/car4_s.txt", "fps": 30, "object_class": "car"},
    {"name": "person1_s", "path": "data_seq/UAV123/person1_s", "startFrame": 1, "endFrame": 1600, "nz": 6,
     "ext": "jpg", "anno_path": "anno/UAV123/person1_s.txt", "fps": 30, "object_class": "person"},
    {"name": "person2_s", "path": "data_seq/UAV123/person2_s", "startFrame": 1, "endFrame": 250, "nz": 6,
     "ext": "jpg", "anno_path": "anno/UAV123/person2_s.txt", "fps": 30, "object_class": "person"},
    {"name": "person3_s", "path": "data_seq/UAV123/person3_s", "startFrame": 1, "endFrame": 505, "nz": 6,
     "ext": "jpg", "anno_path": "anno/UAV123/person3_s.txt", "fps": 30, "object_class": "person"},
]


_uav20l_sequence_info_list = [
    {'name': 'bike1', 'path': 'data_seq/UAV123/bike1', 'startFrame': 1, 'endFrame': 3085, 'nz': 6,
     'ext': 'jpg', 'anno_path': 'anno/UAV20L/bike1.txt', 'fps': 30, "object_class": "vehicle"},
    {'name': 'bird1', 'path': 'data_seq/UAV123/bird1', 'startFrame': 1, 'endFrame': 2437, 'nz': 6,
     'ext': 'jpg', 'anno_path': 'anno/UAV20L/bird1.txt', 'fps': 30, "object_class": "bird"},
    {'name': 'car1', 'path': 'data_seq/UAV123/car1', 'startFrame': 1, 'endFrame': 2629, 'nz': 6,
     'ext': 'jpg', 'anno_path': 'anno/UAV20L/car1.txt', 'fps': 96, "object_class": "car"},
    {'name': 'car3', 'path': 'data_seq/UAV123/car3', 'startFrame': 1, 'endFrame': 1717, 'nz': 6,
     'ext': 'jpg', 'anno_path': 'anno/UAV20L/car3.txt', 'fps': 96, "object_class": "car"},
    {'name': 'car6', 'path': 'data_seq/UAV123/car6', 'startFrame': 1, 'endFrame': 4861, 'nz': 6,
     'ext': 'jpg', 'anno_path': 'anno/UAV20L/car6.txt', 'fps': 96, "object_class": "car"},
    {'name': 'car8', 'path': 'data_seq/UAV123/car8', 'startFrame': 1, 'endFrame': 2575, 'nz': 6,
     'ext': 'jpg', 'anno_path': 'anno/UAV20L/car8.txt', 'fps': 96, "object_class": "car"},
    {'name': 'car9', 'path': 'data_seq/UAV123/car9', 'startFrame': 1, 'endFrame': 1879, 'nz': 6,
     'ext': 'jpg', 'anno_path': 'anno/UAV20L/car9.txt', 'fps': 96, "object_class": "car"},
    {'name': 'car16', 'path': 'data_seq/UAV123/car16', 'startFrame': 1, 'endFrame': 1993, 'nz': 6,
     'ext': 'jpg', 'anno_path': 'anno/UAV20L/car16.txt', 'fps': 96, "object_class": "car"},
    {'name': 'group1', 'path': 'data_seq/UAV123/group1', 'startFrame': 1, 'endFrame': 4873, 'nz': 6,
     'ext': 'jpg', 'anno_path': 'anno/UAV20L/group1.txt', 'fps': 30, "object_class": "person"},
    {'name': 'group2', 'path': 'data_seq/UAV123/group2', 'startFrame': 1, 'endFrame': 2683, 'nz': 6,
     'ext': 'jpg', 'anno_path': 'anno/UAV20L/group2.txt', 'fps': 96, "object_class": "person"},
    {'name': 'group3', 'path': 'data_seq/UAV123/group3', 'startFrame': 1, 'endFrame': 5527, 'nz': 6,
     'ext': 'jpg', 'anno_path': 'anno/UAV20L/group3.txt', 'fps': 96, "object_class": "person"},
    {'name': 'person2', 'path': 'data_seq/UAV123/person2', 'startFrame': 1, 'endFrame': 2623, 'nz': 6,
     'ext': 'jpg', 'anno_path': 'anno/UAV20L/person2.txt', 'fps': 30, "object_class": "person"},
    {'name': 'person4', 'path': 'data_seq/UAV123/person4', 'startFrame': 1, 'endFrame': 2743, 'nz': 6,
     'ext': 'jpg', 'anno_path': 'anno/UAV20L/person4.txt', 'fps': 30, "object_class": "person"},
    {'name': 'person5', 'path': 'data_seq/UAV123/person5', 'startFrame': 1, 'endFrame': 2101, 'nz': 6,
     'ext': 'jpg', 'anno_path': 'anno/UAV20L/person5.txt', 'fps': 30, "object_class": "person"},
    {'name': 'person7', 'path': 'data_seq/UAV123/person7', 'startFrame': 1, 'endFrame': 2065, 'nz': 6,
     'ext': 'jpg', 'anno_path': 'anno/UAV20L/person7.txt', 'fps': 30, "object_class": "person"},
    {'name': 'person7', 'path': 'data_seq/UAV123/person7', 'startFrame': 1, 'endFrame': 2065, 'nz': 6,
     'ext': 'jpg', 'anno_path': 'anno/UAV20L/person7.txt', 'fps': 30, "object_class": "person"},
    {'name': 'person14', 'path': 'data_seq/UAV123/person14', 'startFrame': 1, 'endFrame': 2923, 'nz': 6,
     'ext': 'jpg', 'anno_path': 'anno/UAV20L/person14.txt', 'fps': 96, "object_class": "person"},
    {'name': 'person17', 'path': 'data_seq/UAV123/person17', 'startFrame': 1, 'endFrame': 2347, 'nz': 6,
     'ext': 'jpg', 'anno_path': 'anno/UAV20L/person17.txt', 'fps': 96, "object_class": "person"},
    {'name': 'person19', 'path': 'data_seq/UAV123/person19', 'startFrame': 1, 'endFrame': 4357, 'nz': 6,
     'ext': 'jpg', 'anno_path': 'anno/UAV20L/person19.txt', 'fps': 96, "object_class": "person"},
    {'name': 'person20', 'path': 'data_seq/UAV123/person20', 'startFrame': 1, 'endFrame': 1783, 'nz': 6,
     'ext': 'jpg', 'anno_path': 'anno/UAV20L/person20.txt', 'fps': 96, "object_class": "person"},
    {'name': 'uav1', 'path': 'data_seq/UAV123/uav1', 'startFrame': 1, 'endFrame': 3469, 'nz': 6,
     'ext': 'jpg', 'anno_path': 'anno/UAV20L/uav1.txt', 'fps': 30, "object_class": "aircraft"},
]


def construct_UAV123(constructor: SingleObjectTrackingDatasetConstructor, seed):
    root_path = seed.root_path
    build_UAV20L = seed.build_UAV20L

    if build_UAV20L:
        sequence_info_list = _uav20l_sequence_info_list
        attr_path = os.path.join(root_path, 'anno', 'UAV20L', 'att')
    else:
        sequence_info_list = _uav123_sequence_info_list
        attr_path = os.path.join(root_path, 'anno', 'UAV123', 'att')

    category_id_name_map = {}
    category_name_id_map = {}
    id_ = 0
    for sequence_info in sequence_info_list:
        if sequence_info['object_class'] not in category_name_id_map:
            category_name_id_map[sequence_info['object_class']] = id_
            category_id_name_map[id_] = sequence_info['object_class']
            id_ += 1
    constructor.set_category_id_name_map(category_id_name_map)

    constructor.set_total_number_of_sequences(len(sequence_info_list))

    constructor.set_attribute('attribute short name', ('IV', 'SV', 'POC', 'FOC', 'OV', 'FM', 'CM', 'BC', 'SOB', 'ARC', 'VC', 'LR'))
    constructor.set_attribute('attribute name', ('Illumination Variation', 'Scale Variation', 'Partial Occlusion', 'Full Occlusion', 'Out-of-View', 'Fast Motion', 'Camera Motion', 'Background Clutter', 'Similar Object', 'Aspect Ratio Change', 'Viewpoint Change', 'Low Resolution'))

    for sequence_info in sequence_info_list:
        sequence_path = os.path.join(root_path, sequence_info['path'])
        sequence_annotation_path = os.path.join(root_path, sequence_info['anno_path'])
        category_id = category_name_id_map[sequence_info['object_class']]

        nz = sequence_info['nz']
        ext = sequence_info['ext']
        start_frame = sequence_info['startFrame']
        end_frame = sequence_info['endFrame']

        init_omit = 0
        if 'initOmit' in sequence_info:
            init_omit = sequence_info['initOmit']
        frames = [os.path.join(sequence_path, '{frame:0{nz}}.{ext}'.format(frame=frame_num, nz=nz, ext=ext)) for frame_num in range(start_frame+init_omit, end_frame+1)]
        attr_file_path = os.path.join(attr_path, f'{sequence_info["name"]}.txt')
        sequence_attr = np.loadtxt(attr_file_path, dtype=np.bool_, delimiter=',')

        bounding_boxes = np.loadtxt(sequence_annotation_path, dtype=np.float64, delimiter=',')

        with constructor.new_sequence(category_id) as sequence_constructor:
            sequence_constructor.set_name(sequence_info['name'])
            sequence_constructor.set_fps(sequence_info['fps'])
            '''
            Sequence attributes:
            IV   Illumination Variation: the illumination of the target changes significantly.
            SV   Scale Variation: the ratio of initial and at least one subsequent bounding box is outside the range [0.5, 2].
            POC  Partial Occlusion: the target is partially occluded.
            FOC  Full Occlusion: the target is fully occluded.
            OV   Out-of-View: some portion of the target leaves the view.
            FM   Fast Motion: motion of the ground-truth bounding box is larger than 20 pixels between two consecutive frames.
            CM   Camera Motion: abrupt motion of the camera.
            BC   Background Clutter: the background near the target has similar appearance as the target.
            SOB  Similar Object: there are objects of similar shape or same type near the target.
            ARC  Aspect Ratio Change: the fraction of ground-truth aspect ratio in the first frame and at least one subsequent frame is outside the range [0.5, 2].
            VC   Viewpoint Change: viewpoint affects target appearance significantly.
            LR   Low Resolution: at least one ground-truth bounding box has less than 400 pixels.
            '''
            sequence_constructor.set_attribute('attributes', sequence_attr.tolist())
            sequence_frame_size = _sequence_frame_size[sequence_info['name']]
            for frame_path, bounding_box in zip(frames, bounding_boxes):
                with sequence_constructor.new_frame() as frame_constructor:
                    frame_constructor.set_path(frame_path, sequence_frame_size)
                    if np.isnan(bounding_box).any():
                        frame_constructor.set_bounding_box([0, 0, 0, 0])
                    else:
                        frame_constructor.set_bounding_box(try_get_int_array(bounding_box).tolist())


_sequence_frame_size = {'bike1': [1280, 720], 'bike2': [1280, 720], 'bike3': [1280, 720], 'bird1_1': [1280, 720], 'bird1_2': [1280, 720], 'bird1_3': [1280, 720], 'boat1': [1280, 720], 'boat2': [1280, 720], 'boat3': [1280, 720], 'boat4': [1280, 720], 'boat5': [1280, 720], 'boat6': [1280, 720], 'boat7': [1280, 720], 'boat8': [1280, 720], 'boat9': [1280, 720], 'building1': [1280, 720], 'building2': [1280, 720], 'building3': [1280, 720], 'building4': [1280, 720], 'building5': [1280, 720], 'car10': [1280, 720], 'car11': [1280, 720], 'car12': [1280, 720], 'car13': [1280, 720], 'car14': [1280, 720], 'car15': [1280, 720], 'car16_1': [1280, 720], 'car16_2': [1280, 720], 'car17': [1280, 720], 'car18': [1280, 720], 'car1_1': [1280, 720], 'car1_2': [1280, 720], 'car1_3': [1280, 720], 'car1_s': [1280, 720], 'car2': [1280, 720], 'car2_s': [1280, 720], 'car3': [1280, 720], 'car3_s': [1280, 720], 'car4': [1280, 720], 'car4_s': [1280, 720], 'car5': [1280, 720], 'car6_1': [1280, 720], 'car6_2': [1280, 720], 'car6_3': [1280, 720], 'car6_4': [1280, 720], 'car6_5': [1280, 720], 'car7': [1280, 720], 'car8_1': [1280, 720], 'car8_2': [1280, 720], 'car9': [1280, 720], 'group1_1': [1280, 720], 'group1_2': [1280, 720], 'group1_3': [1280, 720], 'group1_4': [1280, 720], 'group2_1': [1280, 720], 'group2_2': [1280, 720], 'group2_3': [1280, 720], 'group3_1': [1280, 720], 'group3_2': [1280, 720], 'group3_3': [1280, 720], 'group3_4': [1280, 720], 'person1': [1280, 720], 'person10': [1280, 720], 'person11': [1280, 720], 'person12_1': [1280, 720], 'person12_2': [1280, 720], 'person13': [1280, 720], 'person14_1': [1280, 720], 'person14_2': [1280, 720], 'person14_3': [1280, 720], 'person15': [1280, 720], 'person16': [1280, 720], 'person17_1': [1280, 720], 'person17_2': [1280, 720], 'person18': [1280, 720], 'person19_1': [1280, 720], 'person19_2': [1280, 720], 'person19_3': [1280, 720], 'person1_s': [1280, 720], 'person20': [1280, 720], 'person21': [1280, 720], 'person22': [1280, 720], 'person23': [1280, 720], 'person2_1': [1280, 720], 'person2_2': [1280, 720], 'person2_s': [1280, 720], 'person3': [1280, 720], 'person3_s': [1280, 720], 'person4_1': [1280, 720], 'person4_2': [1280, 720], 'person5_1': [1280, 720], 'person5_2': [1280, 720], 'person6': [1280, 720], 'person7_1': [1280, 720], 'person7_2': [1280, 720], 'person8_1': [1280, 720], 'person8_2': [1280, 720], 'person9': [1280, 720], 'truck1': [1280, 720], 'truck2': [1280, 720], 'truck3': [1280, 720], 'truck4_1': [1280, 720], 'truck4_2': [1280, 720], 'uav1_1': [720, 480], 'uav1_2': [720, 480], 'uav1_3': [720, 480], 'uav2': [720, 480], 'uav3': [720, 480], 'uav4': [720, 480], 'uav5': [720, 480], 'uav6': [720, 480], 'uav7': [720, 480], 'uav8': [720, 480], 'wakeboard1': [1280, 720], 'wakeboard10': [1280, 720], 'wakeboard2': [1280, 720], 'wakeboard3': [1280, 720], 'wakeboard4': [1280, 720], 'wakeboard5': [1280, 720], 'wakeboard6': [1280, 720], 'wakeboard7': [1280, 720], 'wakeboard8': [1280, 720], 'wakeboard9': [1280, 720]}
