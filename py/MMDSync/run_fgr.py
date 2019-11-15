
import open3d as o3d
from global_registration import *
import numpy as np
import copy
import sys
import time


def prepare_dataset2(fileName1, fileName2, voxel_size):
    print(":: Load two point clouds and disturb initial pose.")
    #source = o3d.io.read_point_cloud("D:/Data/meas_sync/kbest/ShapeNet2Samples/model0/7c8a7f6ad605c031e8398dbacbe1f3b1/point_cloud/0.obj")
    #target = o3d.io.read_point_cloud("D:/Data/meas_sync/kbest/ShapeNet2Samples/model0/7c8a7f6ad605c031e8398dbacbe1f3b1/point_cloud/12.obj")
    source = o3d.geometry.PointCloud()
    target = o3d.geometry.PointCloud()
    source.points = o3d.io.read_triangle_mesh(fileName1).vertices
    target.points = o3d.io.read_triangle_mesh(fileName2).vertices
    #trans_init = np.asarray([[0.0, 0.0, 1.0, 0.0], [1.0, 0.0, 0.0, 0.0],
     #                        [0.0, 1.0, 0.0, 0.0], [0.0, 0.0, 0.0, 1.0]])
    #source.transform(trans_init)
    #draw_registration_result(source, target, np.identity(4))

    source_down, source_fpfh = preprocess_point_cloud(source, voxel_size)
    target_down, target_fpfh = preprocess_point_cloud(target, voxel_size)

    return source, target, source_down, target_down, source_fpfh, target_fpfh

def execute_fast_global_registration(source_down, target_down, source_fpfh,
                                     target_fpfh, voxel_size):
    distance_threshold = voxel_size * 0.5
    print(":: Apply fast global registration with distance threshold %.3f" \
            % distance_threshold)
    result = o3d.registration.registration_fast_based_on_feature_matching(
        source_down, target_down, source_fpfh, target_fpfh,
        o3d.registration.FastGlobalRegistrationOption(
            maximum_correspondence_distance=distance_threshold))
    return result


if __name__ == "__main__":

    fileName1 = sys.argv[1]
    fileName2 = sys.argv[2]
    outputPose = sys.argv[3]
    voxel_size = 25  # means 5cm for the dataset
    source, target, source_down, target_down, source_fpfh, target_fpfh = \
            prepare_dataset2(fileName1, fileName2, voxel_size)

    start = time.time()
    #result_ransac = execute_global_registration(source_down, target_down,
    #                                            source_fpfh, target_fpfh,
    #                                            voxel_size)
    #print(result_ransac)
    #print("Global registration took %.3f sec.\n" % (time.time() - start))
    #draw_registration_result(source_down, target_down, result_ransac.transformation)
    #print(result_ransac.transformation)

    #start = time.time()
    result_fast = execute_fast_global_registration(source_down, target_down,
                                                   source_fpfh, target_fpfh,
                                                   voxel_size)
    print("Fast global registration took %.3f sec.\n" % (time.time() - start))
    #draw_registration_result(source_down, target_down, result_fast.transformation)

    mat = result_fast.transformation
    #with open(outputPose) as f:
    #    for line in mat:
    np.savetxt(outputPose, mat, fmt='%.4f')