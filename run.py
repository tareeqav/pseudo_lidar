import os
import sys

import numpy as np
import mayavi.mlab as mlab


sys.path.append(f'{os.path.dirname(os.path.realpath(__file__))}/preprocessing')

import kitti_util
import generate_lidar as utils
from viz_util import draw_lidar_simple, draw_lidar, draw_gt_boxes3d


def get_lidar_in_image_fov(pc_velo, calib, xmin, ymin, xmax, ymax,
                           return_more=False, clip_distance=2.0):
    ''' Filter lidar points, keep those in image FOV '''
    pts_2d = calib.project_velo_to_image(pc_velo)
    fov_inds = (pts_2d[:,0]<xmax) & (pts_2d[:,0]>=xmin) & \
        (pts_2d[:,1]<ymax) & (pts_2d[:,1]>=ymin)
    fov_inds = fov_inds & (pc_velo[:,0]>clip_distance)
    imgfov_pc_velo = pc_velo[fov_inds,:]
    if return_more:
        return imgfov_pc_velo, pts_2d, fov_inds
    else:
        return imgfov_pc_velo


def show_lidar_with_boxes(pc_velo, calib,
                          img_fov=False, img_width=None, img_height=None): 
      ''' Show all LiDAR points.
         Draw 3d box in LiDAR point cloud (in velo coord system) '''

      print(('All point num: ', pc_velo.shape[0]))
      fig = mlab.figure(figure=None, bgcolor=(0,0,0),
         fgcolor=None, engine=None, size=(1000, 500))
      if img_fov:
         pc_velo = get_lidar_in_image_fov(pc_velo, calib, 0, 0,
            img_width, img_height)
         print(('FOV point num: ', pc_velo.shape[0]))
      draw_lidar(pc_velo, fig=fig)

      # for obj in objects:
      #    if obj.type=='DontCare':continue
      #    # Draw 3d bounding box
      #    box3d_pts_2d, box3d_pts_3d = utils.compute_box_3d(obj, calib.P) 
      #    box3d_pts_3d_velo = calib.project_rect_to_velo(box3d_pts_3d)
      #    # Draw heading arrow
      #    ori3d_pts_2d, ori3d_pts_3d = utils.compute_orientation_3d(obj, calib.P)
      #    ori3d_pts_3d_velo = calib.project_rect_to_velo(ori3d_pts_3d)
      #    x1,y1,z1 = ori3d_pts_3d_velo[0,:]
      #    x2,y2,z2 = ori3d_pts_3d_velo[1,:]
      #    draw_gt_boxes3d([box3d_pts_3d_velo], fig=fig)
      #    mlab.plot3d([x1, x2], [y1, y2], [z1,z2], color=(0.5,0.5,0.5),
      #       tube_radius=None, line_width=1, figure=fig)
      mlab.show(1)
      input()


def call_show_lidar_with_boxes(disp_map, calib_filepath):
   # display point cloud
   calib = kitti_util.Calibration(calib_filepath)
   lidar = utils.project_disp_to_points(calib, disp_map, 1)
   show_lidar_with_boxes(lidar, calib, img_width=1224, img_height=370)


def run(disp_map, calib_filepath, image_filename, max_high=1):
      """
      """

      disp_map = (disp_map*256).astype(np.uint16)/256.
      calib = kitti_util.Calibration(calib_filepath)

      lidar = utils.project_disp_to_points(calib, disp_map, max_high)
      # pad 1 in the indensity dimension
      lidar = np.concatenate([lidar, np.ones((lidar.shape[0], 1))], 1)
      
      print('>>>>> generated LIDAR has shape', lidar.shape)
      # save output
      lidar = lidar.astype(np.float32)
      lidar.tofile(f'{os.path.dirname(os.path.realpath(__file__))}/output/{image_filename}.bin')
      print('saved file',f'{os.path.dirname(os.path.realpath(__file__))}/output/{image_filename}.bin')
      # display point cloud
      # show_lidar_with_boxes(lidar, calib, img_width=1224, img_height=370)
