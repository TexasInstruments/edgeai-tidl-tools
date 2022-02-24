import numpy as np
import cv2

def voxelization(lidar_data=[], params=[], voxel_data=[], indices=[], scale_fact = 32.0):

    scratch_1 =[]
    scratch_2 =[]
    enable_pre_proc = True

    if enable_pre_proc == True:
        for i, data in enumerate(lidar_data):

            x = data[0]
            y = data[1]
            z = data[2]

            if ((x > params['min_x']) and (x < params['max_x']) and (y > params['min_y']) and (y < params['max_y']) and
                (z > params['min_z']) and (z < params['max_z'])):

                x_id = (int)(((x - params['min_x']) / params['voxel_size_x']))
                y_id = (int)(((y - params['min_y']) / params['voxel_size_y']))
                scratch_1.append(y_id * params['num_voxel_x'] + x_id)
            else:
                scratch_1.append(-1 - i) # filing unique non valid index

        num_points = np.zeros(params['nw_max_num_voxels'],dtype=int)

        # Find unique indices
        # There will be voxel which doesnt have any 3d point, hence collecting the voxel ids for valid voxels*/
        # scratch_2 is the index in valid voxels
        num_non_empty_voxels = 0
        for i in range(len(lidar_data)):
            if (scratch_1[i] >= 0):

                find_voxel = scratch_1[i] in scratch_1[:i]

                if find_voxel == False:
                    scratch_2.append(num_non_empty_voxels) # this voxel idx has come first time, hence allocate a new index for this
                    indices[0][num_non_empty_voxels] = scratch_1[i]
                    num_non_empty_voxels += 1
                else:
                    k = scratch_1[:i].index(scratch_1[i])
                    scratch_2.append(scratch_2[k]) #already this voxel is having one id hence reuse it
            else:
                scratch_2.append(None)

        #Even though current_voxels is less than params['nw_max_num_voxels'], then also arrange
        #    the data as per maximum number of voxels.

        line_pitch = params['nw_max_num_voxels']
        channel_pitch = params['max_points_per_voxel'] * line_pitch
        j = 0
        tot_num_pts = 0

        for i in range(len(lidar_data)):
            if (scratch_1[i] >= 0):
                j = scratch_2[i] #voxel index
                if(num_points[j]<params['max_points_per_voxel']):
                    voxel_data[0][num_points[j]][j] = lidar_data[i][0] * scale_fact
                    voxel_data[1][num_points[j]][j] = lidar_data[i][1] * scale_fact
                    voxel_data[2][num_points[j]][j] = lidar_data[i][2] * scale_fact
                    voxel_data[3][num_points[j]][j] = lidar_data[i][3] * scale_fact
                    num_points[j] = num_points[j] + 1
                else:
                    tot_num_pts = tot_num_pts+1

        line_pitch = params['nw_max_num_voxels']
        channel_pitch = params['max_points_per_voxel'] * line_pitch
        x_offset = params['voxel_size_x'] / 2 + params['min_x']
        y_offset = params['voxel_size_y'] / 2 + params['min_y']

        for i in range(num_non_empty_voxels):
            x = 0
            y = 0
            z = 0

            for j in range(num_points[i]):
                x += voxel_data[0][j][i]
                y += voxel_data[1][j][i]
                z += voxel_data[2][j][i]

            x_avg = x / num_points[i]
            y_avg = y / num_points[i]
            z_avg = z / num_points[i]

            voxel_center_y = (int)(indices[0][i] / params['num_voxel_x'])
            voxel_center_x = (int)(indices[0][i] - ((int)(voxel_center_y)) * params['num_voxel_x'])


            voxel_center_x *= params['voxel_size_x']
            voxel_center_x += x_offset

            voxel_center_y *= params['voxel_size_y']
            voxel_center_y += y_offset

            for j in range(num_points[i]):
                voxel_data[4][j][i] = voxel_data[0][j][i] - x_avg
                voxel_data[5][j][i] = voxel_data[1][j][i] - y_avg
                voxel_data[6][j][i] = voxel_data[2][j][i] - z_avg
                voxel_data[7][j][i] = voxel_data[0][j][i] - voxel_center_x * scale_fact
                voxel_data[8][j][i] = voxel_data[1][j][i] - voxel_center_y * scale_fact

            #/*looks like bug in python mmdetection3d code, hence below code is to mimic the mmdetect behaviour*/
            for j in range (num_points[i]):
                voxel_data[0][j][i] = voxel_data[7][j][i]
                voxel_data[1][j][i] = voxel_data[8][j][i]

        # Number of points in each voxel is not given to algorithm, here '-1' acts as marker position, as zero is valid entry
        indices[0][num_non_empty_voxels] = -1
        indices[1:64] = indices[0]

        voxel_data = voxel_data.astype("int32")
        voxel_data = voxel_data.astype("float32")

    else:

        voxel_data = np.fromfile("voxel_input0_f32.bin", dtype='float32')
        indices = np.fromfile("indices_input2_f32.bin", dtype='float32')

        voxel_data = input0.astype("int32")
        voxel_data = input0.astype("float32")

        voxel_data = input0.reshape(1, 9, params['max_points_per_voxel'], params['nw_max_num_voxels'])
        indices = input2.reshape(1, 64, params['nw_max_num_voxels']).astype('int32')


#https://github.com/open-mmlab/mmdetection3d/
def boxes3d_to_corners3d_lidar(boxes3d, bottom_center=True):
    """Convert kitti center boxes to corners.

        7 -------- 4
       /|         /|
      6 -------- 5 .
      | |        | |
      . 3 -------- 0
      |/         |/
      2 -------- 1

    Args:
        boxes3d (np.ndarray): Boxes with shape of (N, 7) \
            [x, y, z, w, l, h, ry] in LiDAR coords, see the definition of ry \
            in KITTI dataset.
        bottom_center (bool): Whether z is on the bottom center of object.

    Returns:
        np.ndarray: Box corners with the shape of [N, 8, 3].
    """
    boxes_num = boxes3d.shape[0]
    w, l, h = boxes3d[:, 3], boxes3d[:, 4], boxes3d[:, 5]
    x_corners = np.array(
        [w / 2., -w / 2., -w / 2., w / 2., w / 2., -w / 2., -w / 2., w / 2.],
        dtype=np.float32).T
    y_corners = np.array(
        [-l / 2., -l / 2., l / 2., l / 2., -l / 2., -l / 2., l / 2., l / 2.],
        dtype=np.float32).T

    if bottom_center:
        z_corners = np.zeros((boxes_num, 8), dtype=np.float32)
        z_corners[:, 4:8] = h.reshape(boxes_num, 1).repeat(4, axis=1)  # (N, 8)
    else:
        z_corners = np.array([
            -h / 2., -h / 2., -h / 2., -h / 2., h / 2., h / 2., h / 2., h / 2.
        ],
                             dtype=np.float32).T

    ry = boxes3d[:, 6]
    zeros, ones = np.zeros(
        ry.size, dtype=np.float32), np.ones(
            ry.size, dtype=np.float32)
    rot_list = np.array([[np.cos(ry), -np.sin(ry), zeros],
                         [np.sin(ry), np.cos(ry), zeros], [zeros, zeros,
                                                           ones]])  # (3, 3, N)
    R_list = np.transpose(rot_list, (2, 0, 1))  # (N, 3, 3)

    temp_corners = np.concatenate((x_corners.reshape(
        -1, 8, 1), y_corners.reshape(-1, 8, 1), z_corners.reshape(-1, 8, 1)),
                                  axis=2)  # (N, 8, 3)
    rotated_corners = np.matmul(temp_corners, R_list)  # (N, 8, 3)
    x_corners = rotated_corners[:, :, 0]
    y_corners = rotated_corners[:, :, 1]
    z_corners = rotated_corners[:, :, 2]

    x_loc, y_loc, z_loc = boxes3d[:, 0], boxes3d[:, 1], boxes3d[:, 2]

    x = x_loc.reshape(-1, 1) + x_corners.reshape(-1, 8)
    y = y_loc.reshape(-1, 1) + y_corners.reshape(-1, 8)
    z = z_loc.reshape(-1, 1) + z_corners.reshape(-1, 8)

    corners = np.concatenate(
        (x.reshape(-1, 8, 1), y.reshape(-1, 8, 1), z.reshape(-1, 8, 1)),
        axis=2)

    return corners.astype(np.float32)

#https://github.com/open-mmlab/mmdetection3d/
def draw_lidar_bbox3d_on_img(corners_3d,
                             raw_img,
                             lidar2img_rt,
                             color=(0, 255, 0),
                             thickness=1):
    """Project the 3D bbox on 2D plane and draw on input image.

    Args:
        bboxes3d (numpy.array, shape=[M, 7]):
            3d bbox (x, y, z, dx, dy, dz, yaw) to visualize.
        raw_img (numpy.array): The numpy array of image.
        lidar2img_rt (numpy.array, shape=[4, 4]): The projection matrix
            according to the camera intrinsic parameters.
        color (tuple[int]): The color to draw bboxes. Default: (0, 255, 0).
        thickness (int, optional): The thickness of bboxes. Default: 1.
    """
    img = raw_img.copy()
    num_bbox = corners_3d.shape[0]
    pts_4d = np.concatenate(
        [corners_3d.reshape(-1, 3),
         np.ones((num_bbox * 8, 1))], axis=-1)

    pts_2d = pts_4d @ lidar2img_rt.T

    pts_2d[:, 2] = np.clip(pts_2d[:, 2], a_min=1e-5, a_max=1e5)
    pts_2d[:, 0] /= pts_2d[:, 2]
    pts_2d[:, 1] /= pts_2d[:, 2]
    imgfov_pts_2d = pts_2d[..., :2].reshape(num_bbox, 8, 2)

    line_indices = ((0, 1), (0, 3), (0, 4), (1, 2), (1, 5), (3, 2), (3, 7),
                    (4, 5), (4, 7), (2, 6), (5, 6), (6, 7))
    for i in range(num_bbox):
        corners = imgfov_pts_2d[i].astype(np.int)
        for start, end in line_indices:
            cv2.line(img, (corners[start, 0], corners[start, 1]),
                     (corners[end, 0], corners[end, 1]), color, thickness,
                     cv2.LINE_AA)

    return img.astype(np.uint8)

#https://github.com/dtczhl/dtc-KITTI-For-Beginners/
def load_velodyne_points(filename):
    points = np.fromfile(filename, dtype=np.float32).reshape(-1, 4)
    # points = points[:, :3]  # exclude luminance
    return points

CAM = 2

#https://github.com/dtczhl/dtc-KITTI-For-Beginners/
def load_calib(calib_dir_lines):
    # P2 * R0_rect * Tr_velo_to_cam * y
    #lines = open(calib_dir).readlines()
    lines = [line.split()[1:] for line in calib_dir_lines][:-1]
    #
    P = np.array(lines[CAM]).reshape(3, 4)
    #
    Tr_velo_to_cam = np.array(lines[5]).reshape(3, 4)
    Tr_velo_to_cam = np.concatenate([Tr_velo_to_cam, np.array([0, 0, 0, 1]).reshape(1, 4)], 0)
    #
    R_cam_to_rect = np.eye(4)
    R_cam_to_rect[:3, :3] = np.array(lines[4][:9]).reshape(3, 3)
    #
    P = P.astype('float32')
    Tr_velo_to_cam = Tr_velo_to_cam.astype('float32')
    R_cam_to_rect = R_cam_to_rect.astype('float32')
    return P, Tr_velo_to_cam, R_cam_to_rect

#https://github.com/dtczhl/dtc-KITTI-For-Beginners/
def prepare_velo_points(pts3d_raw):
    '''Replaces the reflectance value by 1, and tranposes the array, so
        points can be directly multiplied by the camera projection matrix'''
    pts3d = pts3d_raw
    # Reflectance > 0
    indices = pts3d[:, 3] > 0
    pts3d = pts3d[indices, :]
    pts3d[:, 3] = 1
    return pts3d.transpose(), indices

#https://github.com/dtczhl/dtc-KITTI-For-Beginners/
def project_velo_points_in_img(pts3d, T_cam_velo, Rrect, Prect):
    '''Project 3D points into 2D image. Expects pts3d as a 4xN
        numpy array. Returns the 2D projection of the points that
        are in front of the camera only an the corresponding 3D points.'''
    # 3D points in camera reference frame.
    pts3d_cam = Rrect.dot(T_cam_velo.dot(pts3d))
    # Before projecting, keep only points with z>0
    # (points that are in fronto of the camera).
    idx = (pts3d_cam[2, :] >= 0)
    pts2d_cam = Prect.dot(pts3d_cam[:, idx])
    return pts3d[:, idx], pts2d_cam / pts2d_cam[2, :], idx

#https://github.com/dtczhl/dtc-KITTI-For-Beginners/
def align_img_and_pc(img, pts, calib_dir_lines):
    #img = imageio.imread(img_dir)
    #img = cv2.imread(img_dir)
    #pts = load_velodyne_points(pc_dir)
    P, Tr_velo_to_cam, R_cam_to_rect = load_calib(calib_dir_lines)

    pts3d, indices = prepare_velo_points(pts)
    # pts3d_ori = pts3d.copy()
    reflectances = pts[indices, 3]
    pts3d, pts2d_normed, idx = project_velo_points_in_img(pts3d, Tr_velo_to_cam, R_cam_to_rect, P)
    # print reflectances.shape, idx.shape
    reflectances = reflectances[idx]
    # print reflectances.shape, pts3d.shape, pts2d_normed.shape
    # assert reflectances.shape[0] == pts3d.shape[1] == pts2d_normed.shape[1]

    rows, cols = img.shape[:2]

    points = []
    for i in range(pts2d_normed.shape[1]):
        c = int(np.round(pts2d_normed[0, i]))
        r = int(np.round(pts2d_normed[1, i]))
        if c < cols and r < rows and r > 0 and c > 0:
            #color = img[r, c, :]
            #point = [pts3d[0, i], pts3d[1, i], pts3d[2, i], reflectances[i], color[0], color[1], color[2],
            #         pts2d_normed[0, i], pts2d_normed[1, i]]
            point = [pts3d[0, i], pts3d[1, i], pts3d[2, i], reflectances[i]]
            points.append(point)

    points = np.array(points)
    lidr_2_img = P @ R_cam_to_rect @ Tr_velo_to_cam

    return points, lidr_2_img
