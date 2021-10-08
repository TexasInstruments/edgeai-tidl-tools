import math
import cv2
import numpy as np
from scripts.group import HeatmapParser

def define_cfg(udp=False):
    
    cfg = {}

    cfg['higher_hr'] = False
    if not(udp):
        cfg['project2image'] = True
        cfg['use_udp'] = False
    else:
        cfg['project2image'] = False
        cfg['use_udp'] = True       

    cfg['num_joints'] = 17
    cfg['max_num_people'] = 30
    cfg['with_heatmaps'] = [True, True]
    cfg['with_ae'] = [True, False]
    cfg['tag_per_joint'] = True
    cfg['flip_index'] = [0, 2, 1, 4, 3, 6, 5, 8, 7, 10, 9, 12, 11, 14, 13, 16, 15]

    cfg['s'] = 1
    cfg['test_scale_factor'] = [1]
    cfg['flip_test'] = False

    cfg['adjust'] = True
    cfg['refine'] = True

    cfg['detection_threshold'] = 0.1
    cfg['tag_threshold'] = 1
    cfg['use_detection_val'] = True
    cfg['ignore_too_much'] = False

    cfg['nms_kernel'] = 5
    cfg['nms_padding'] = 2
    
    return cfg


# def _ceil_to_multiples_of(x, base=64):
#     """Transform x to the integral multiple of the base."""
#     return int(np.ceil(x / base)) * base

# def _get_multi_scale_size(image_to_read,
#                           input_size,
#                           current_scale,
#                           min_scale):
#     """Get the size for multi-scale training.

#     Args:
#         image: Input image.
#         input_size (int): Size of the image input.
#         current_scale (float): Scale factor.
#         min_scale (float): Minimal scale.
#         use_udp (bool): To use unbiased data processing.
#             Paper ref: Huang et al. The Devil is in the Details: Delving into
#             Unbiased Data Processing for Human Pose Estimation (CVPR 2020).

#     Returns:
#         tuple: A tuple containing multi-scale sizes.

#         - (w_resized, h_resized) (tuple(int)): resized width/height
#         - center (np.ndarray)image center
#         - scale (np.ndarray): scales wrt width/height
#     """

#     image = mmcv.imread(image_to_read)
#     h, w, _ = image.shape

#     # calculate the size for min_scale
#     min_input_size = _ceil_to_multiples_of(min_scale * input_size, 64)
#     if w < h:
#         w_resized = int(min_input_size * current_scale / min_scale)
#         h_resized = int(
#             _ceil_to_multiples_of(min_input_size / w * h, 64) * current_scale /
#             min_scale)
#         if cfg['use_udp']:
#             scale_w = w - 1.0
#             scale_h = (h_resized - 1.0) / (w_resized - 1.0) * (w - 1.0)
#         else:
#             scale_w = w / 200.0
#             scale_h = h_resized / w_resized * w / 200.0
#     else:
#         h_resized = int(min_input_size * current_scale / min_scale)
#         w_resized = int(
#             _ceil_to_multiples_of(min_input_size / h * w, 64) * current_scale /
#             min_scale)
#         if cfg['use_udp']:
#             scale_h = h - 1.0
#             scale_w = (w_resized - 1.0) / (h_resized - 1.0) * (h - 1.0)
#         else:
#             scale_h = h / 200.0
#             scale_w = w_resized / h_resized * h / 200.0
#     if cfg['use_udp']:
#         center = (scale_w / 2.0, scale_h / 2.0)
#     else:
#         center = np.array([round(w / 2.0), round(h / 2.0)])

#     # print("\n")
#     # print("h, w : {}, {}".format(h,w))
#     # print("min_scale,input_size: {}, {}".format(min_scale,input_size))
#     # print("min_input_size : {}".format(min_input_size))
#     # print("h_resized, w_resized: {},{}".format(h_resized, w_resized))
#     # print("center : {}".format(center))
#     # print("scale_h, scale_w : {}, {}".format(scale_h, scale_w))


#     return (w_resized, h_resized), center, np.array([scale_w, scale_h])

def get_multi_stage_outputs(outputs,
                            outputs_flip,
                            num_joints,
                            with_heatmaps,
                            with_ae,
                            tag_per_joint=True,
                            flip_index=None,
                            project2image=True,
                            size_projected=None,
                            align_corners=False):
    """Inference the model to get multi-stage outputs (heatmaps & tags), and
    resize them to base sizes.
    Also to aggregate them.
    """
    
    heatmaps_avg = 0
    heatmaps = []
    tags = []
    
    aggregated_heatmaps = None
    tags_list = []

    flip_test = outputs_flip is not None
    
    offset_feat = num_joints
    
    heatmaps_avg += outputs[0][:, :num_joints]
    tags.append(outputs[0][:, offset_feat:])
    
    heatmaps.append(heatmaps_avg)
        
    if flip_test and flip_index:
        # perform flip testing
        heatmaps_avg = 0        
        offset_feat = num_joints
        
        heatmaps_avg += outputs_flip[0][:, :num_joints][:, flip_index, :, :]
        tags.append(outputs_flip[0][:, offset_feat:])
        if tag_per_joint:
            tags[-1] = tags[-1][:, flip_index, :, :]
         
        heatmaps.append(heatmaps_avg)

    #align corners on the basis of udp, if udp true then align
    #remember project2image is true mostly when udp is False
    if project2image and size_projected:
        
        dim = (size_projected[1], size_projected[0])

        final_heatmaps =[]
        final_tags = []
        
        new_heatmaps = np.empty((0,dim[0],dim[1]),int)
        for hms in heatmaps[0][0]:
            new_hms = cv2.resize(
                hms,
                dim,
                interpolation=cv2.INTER_LINEAR)
            new_heatmaps = np.append(new_heatmaps,[new_hms],axis=0)
        
        final_heatmaps.append(np.expand_dims(new_heatmaps,0))
        
        if flip_test:
            new_heatmaps_flipped = np.empty((0,dim[0],dim[1]),int)
            for hms in heatmaps[1][0]:
                new_hms = cv2.resize(
                    hms,
                    dim,
                    interpolation=cv2.INTER_LINEAR)
                new_heatmaps_flipped = np.append(new_heatmaps_flipped,[new_hms],axis=0)
                
            final_heatmaps.append(np.expand_dims(new_heatmaps_flipped,0))
        
        new_tags = np.empty((0,dim[0],dim[1]),int)
        for tms in tags[0][0]:
            new_tms =  cv2.resize(
                tms,
                dim,
                interpolation=cv2.INTER_LINEAR)
            new_tags = np.append(new_tags,[new_tms],axis=0)
            
            
        final_tags.append(np.expand_dims(new_tags,0))
        
        if flip_test:
            new_tags_flipped = np.empty((0,dim[0],dim[1]),int)
            for tms in tags[1][0]:
                new_tms = cv2.resize(
                    tms,
                    dim,
                    interpolation=cv2.INTER_LINEAR)
                new_tags_flipped = np.append(new_tags_flipped,[new_tms],axis=0)
                
            final_tags.append(np.expand_dims(new_tags_flipped,0))
            
    else:
        final_tags = tags
        final_heatmaps = heatmaps
            
    for tms in final_tags:
        tags_list.append(np.expand_dims(tms,axis=4))
        
    aggregated_heatmaps = (final_heatmaps[0] +
                    final_heatmaps[1]) / 2.0 if flip_test else final_heatmaps[0] 
    
    tags = np.concatenate(tags_list,axis=4)
        
    return aggregated_heatmaps, tags


def transform_preds(coords, center, scale, output_size, use_udp=False):
    """Get final keypoint predictions from heatmaps and apply scaling and
    translation to map them back to the image.

    Note:
        num_keypoints: K

    Args:
        coords (np.ndarray[K, ndims]):

            * If ndims=2, corrds are predicted keypoint location.
            * If ndims=4, corrds are composed of (x, y, scores, tags)
            * If ndims=5, corrds are composed of (x, y, scores, tags,
              flipped_tags)

        center (np.ndarray[2, ]): Center of the bounding box (x, y).
        scale (np.ndarray[2, ]): Scale of the bounding box
            wrt [width, height].
        output_size (np.ndarray[2, ] | list(2,)): Size of the
            destination heatmaps.
        use_udp (bool): Use unbiased data processing

    Returns:
        np.ndarray: Predicted coordinates in the images.
    """
    assert coords.shape[1] in (2, 4, 5)
    assert len(center) == 2
    assert len(scale) == 2
    assert len(output_size) == 2

    # Recover the scale which is normalized by a factor of 200.
    scale = scale * 200.0

    if use_udp:
        scale_x = scale[0] / (output_size[0] - 1.0)
        scale_y = scale[1] / (output_size[1] - 1.0)
    else:
        scale_x = scale[0] / output_size[0]
        scale_y = scale[1] / output_size[1]

    target_coords = np.ones_like(coords)
    target_coords[:, 0] = coords[:, 0] * scale_x + center[0] - scale[0] * 0.5
    target_coords[:, 1] = coords[:, 1] * scale_y + center[1] - scale[1] * 0.5

    return target_coords

def get_warp_matrix(theta, size_input, size_dst, size_target):
    """Calculate the transformation matrix under the constraint of unbiased.
    Paper ref: Huang et al. The Devil is in the Details: Delving into Unbiased
    Data Processing for Human Pose Estimation (CVPR 2020).

    Args:
        theta (float): Rotation angle in degrees.
        size_input (np.ndarray): Size of input image [w, h].
        size_dst (np.ndarray): Size of output image [w, h].
        size_target (np.ndarray): Size of ROI in input plane [w, h].

    Returns:
        matrix (np.ndarray): A matrix for transformation.
    """
    theta = np.deg2rad(theta)
    matrix = np.zeros((2, 3), dtype=np.float32)
    scale_x = size_dst[0] / size_target[0]
    scale_y = size_dst[1] / size_target[1]
    matrix[0, 0] = math.cos(theta) * scale_x
    matrix[0, 1] = -math.sin(theta) * scale_x
    matrix[0, 2] = scale_x * (-0.5 * size_input[0] * math.cos(theta) +
                              0.5 * size_input[1] * math.sin(theta) +
                              0.5 * size_target[0])
    matrix[1, 0] = math.sin(theta) * scale_y
    matrix[1, 1] = math.cos(theta) * scale_y
    matrix[1, 2] = scale_y * (-0.5 * size_input[0] * math.sin(theta) -
                              0.5 * size_input[1] * math.cos(theta) +
                              0.5 * size_target[1])
    return matrix

def warp_affine_joints(joints, mat):
    """Apply affine transformation defined by the transform matrix on the
    joints.

    Args:
        joints (np.ndarray[..., 2]): Origin coordinate of joints.
        mat (np.ndarray[3, 2]): The affine matrix.

    Returns:
        matrix (np.ndarray[..., 2]): Result coordinate of joints.
    """
    joints = np.array(joints)
    shape = joints.shape
    joints = joints.reshape(-1, 2)
    return np.dot(
        np.concatenate((joints, joints[:, 0:1] * 0 + 1), axis=1),
        mat.T).reshape(shape)

def get_group_preds(grouped_joints,
                    center,
                    scale,
                    heatmap_size,
                    use_udp=False):
    """Transform the grouped joints back to the image.

    Args:
        grouped_joints (list): Grouped person joints.
        center (np.ndarray[2, ]): Center of the bounding box (x, y).
        scale (np.ndarray[2, ]): Scale of the bounding box
            wrt [width, height].
        heatmap_size (np.ndarray[2, ]): Size of the destination heatmaps.
        use_udp (bool): Unbiased data processing.
             Paper ref: Huang et al. The Devil is in the Details: Delving into
             Unbiased Data Processing for Human Pose Estimation (CVPR 2020).

    Returns:
        list: List of the pose result for each person.
    """
    if use_udp:
        if grouped_joints[0].shape[0] > 0:
            heatmap_size_t = np.array(heatmap_size, dtype=np.float32) - 1.0
            trans = get_warp_matrix(
                theta=0,
                size_input=heatmap_size_t,
                size_dst=scale,
                size_target=heatmap_size_t)
            grouped_joints[0][..., :2] = \
                warp_affine_joints(grouped_joints[0][..., :2], trans)
        results = [person for person in grouped_joints[0]]
    else:
        results = []
        for person in grouped_joints[0]:
            joints = transform_preds(person, center, scale, heatmap_size)
            results.append(joints)
    return results


def oks_iou(g, d, a_g, a_d, sigmas=None, vis_thr=None):
    """Calculate oks ious.

    Args:
        g: Ground truth keypoints.
        d: Detected keypoints.
        a_g: Area of the ground truth object.
        a_d: Area of the detected object.
        sigmas: standard deviation of keypoint labelling.
        vis_thr: threshold of the keypoint visibility.

    Returns:
        list: The oks ious.
    """
    if sigmas is None:
        sigmas = np.array([
            .26, .25, .25, .35, .35, .79, .79, .72, .72, .62, .62, 1.07, 1.07,
            .87, .87, .89, .89
        ]) / 10.0
    vars = (sigmas * 2)**2
    xg = g[0::3]
    yg = g[1::3]
    vg = g[2::3]
    ious = np.zeros(len(d), dtype=np.float32)
    for n_d in range(0, len(d)):
        xd = d[n_d, 0::3]
        yd = d[n_d, 1::3]
        vd = d[n_d, 2::3]
        dx = xd - xg
        dy = yd - yg
        e = (dx**2 + dy**2) / vars / ((a_g + a_d[n_d]) / 2 + np.spacing(1)) / 2
        if vis_thr is not None:
            ind = list(vg > vis_thr) and list(vd > vis_thr)
            e = e[ind]
        ious[n_d] = np.sum(np.exp(-e)) / len(e) if len(e) != 0 else 0.0
    return ious

def oks_nms(kpts_db, thr, sigmas=None, vis_thr=None):
    """OKS NMS implementations.

    Args:
        kpts_db: keypoints.
        thr: Retain overlap < thr.
        sigmas: standard deviation of keypoint labelling.
        vis_thr: threshold of the keypoint visibility.

    Returns:
        np.ndarray: indexes to keep.
    """
    if len(kpts_db) == 0:
        return []

    scores = np.array([k['score'] for k in kpts_db])
    kpts = np.array([k['keypoints'].flatten() for k in kpts_db])
    areas = np.array([k['area'] for k in kpts_db])

    order = scores.argsort()[::-1]

    keep = []
    while len(order) > 0:
        i = order[0]
        keep.append(i)

        oks_ovr = oks_iou(kpts[i], kpts[order[1:]], areas[i], areas[order[1:]],
                          sigmas, vis_thr)

        inds = np.where(oks_ovr <= thr)[0]
        order = order[inds + 1]

    keep = np.array(keep)

    return keep

def show_result(img,
                result,
                skeleton=None,
                kpt_score_thr=0.3,
                bbox_color=None,
                pose_kpt_color=None,
                pose_limb_color=None,
                radius=4,
                thickness=1,
                font_scale=0.5,
                win_name='',
                show=False,
                show_keypoint_weight=False,
                wait_time=0,
                out_file=None):
    """Draw `result` over `img`.

    Args:
        img (str or Tensor): The image to be displayed.
        result (list[dict]): The results to draw over `img`
            (bbox_result, pose_result).
        skeleton (list[list]): The connection of keypoints.
        kpt_score_thr (float, optional): Minimum score of keypoints
            to be shown. Default: 0.3.
        pose_kpt_color (np.array[Nx3]`): Color of N keypoints.
            If None, do not draw keypoints.
        pose_limb_color (np.array[Mx3]): Color of M limbs.
            If None, do not draw limbs.
        radius (int): Radius of circles.
        thickness (int): Thickness of lines.
        font_scale (float): Font scales of texts.
        win_name (str): The window name.
        show (bool): Whether to show the image. Default: False.
        show_keypoint_weight (bool): Whether to change the transparency
            using the predicted confidence scores of keypoints.
        wait_time (int): Value of waitKey param.
            Default: 0.
        out_file (str or None): The filename to write the image.
            Default: None.

    Returns:
        Tensor: Visualized image only if not `show` or `out_file`
    """

    img = cv2.imread(img)
    img = img[:,:,::-1]
    img = img.copy()
    img_h, img_w, _ = img.shape

    pose_result = []
    for res in result:
        pose_result.append(res['keypoints'])

    for _, kpts in enumerate(pose_result):
        # draw each point on image
        if pose_kpt_color is not None:
            assert len(pose_kpt_color) == len(kpts)
            for kid, kpt in enumerate(kpts):
                x_coord, y_coord, kpt_score = int(kpt[0]), int(
                    kpt[1]), kpt[2]
                if kpt_score > kpt_score_thr:
                    if show_keypoint_weight:
                        img_copy = img.copy()
                        r, g, b = pose_kpt_color[kid]
                        cv2.circle(img_copy, (int(x_coord), int(y_coord)),
                                   radius, (int(r), int(g), int(b)), -1)
                        transparency = max(0, min(1, kpt_score))
                        cv2.addWeighted(
                            img_copy,
                            transparency,
                            img,
                            1 - transparency,
                            0,
                            dst=img)
                    else:
                        r, g, b = pose_kpt_color[kid]
                        cv2.circle(img, (int(x_coord), int(y_coord)),
                                   radius, (int(r), int(g), int(b)), -1)

        # draw limbs
        if skeleton is not None and pose_limb_color is not None:
            assert len(pose_limb_color) == len(skeleton)
            for sk_id, sk in enumerate(skeleton):
                pos1 = (int(kpts[sk[0] - 1, 0]), int(kpts[sk[0] - 1, 1]))
                pos2 = (int(kpts[sk[1] - 1, 0]), int(kpts[sk[1] - 1, 1]))
                if (pos1[0] > 0 and pos1[0] < img_w and pos1[1] > 0
                        and pos1[1] < img_h and pos2[0] > 0
                        and pos2[0] < img_w and pos2[1] > 0
                        and pos2[1] < img_h
                        and kpts[sk[0] - 1, 2] > kpt_score_thr
                        and kpts[sk[1] - 1, 2] > kpt_score_thr):
                    r, g, b = pose_limb_color[sk_id]
                    if show_keypoint_weight:
                        img_copy = img.copy()
                        X = (pos1[0], pos2[0])
                        Y = (pos1[1], pos2[1])
                        mX = np.mean(X)
                        mY = np.mean(Y)
                        length = ((Y[0] - Y[1])**2 + (X[0] - X[1])**2)**0.5
                        angle = math.degrees(
                            math.atan2(Y[0] - Y[1], X[0] - X[1]))
                        stickwidth = 2
                        polygon = cv2.ellipse2Poly(
                            (int(mX), int(mY)),
                            (int(length / 2), int(stickwidth)), int(angle),
                            0, 360, 1)
                        cv2.fillConvexPoly(img_copy, polygon,
                                           (int(r), int(g), int(b)))
                        transparency = max(
                            0,
                            min(
                                1, 0.5 *
                                (kpts[sk[0] - 1, 2] + kpts[sk[1] - 1, 2])))
                        cv2.addWeighted(
                            img_copy,
                            transparency,
                            img,
                            1 - transparency,
                            0,
                            dst=img)
                    else:
                        cv2.line(
                            img,
                            pos1,
                            pos2, (int(r), int(g), int(b)),
                            thickness=thickness)

    if show:
        imshow(img, win_name, wait_time)

    if out_file is not None:
        imwrite(img, out_file)

    return img

def vis_pose_result(img,
                    result,
                    kpt_score_thr=0.3,
                    show=False,
                    out_file=None,
                    thickness=1,
                    radius=4):
    
    palette = np.array([[255, 128, 0], [255, 153, 51], [255, 178, 102],
                        [230, 230, 0], [255, 153, 255], [153, 204, 255],
                        [255, 102, 255], [255, 51, 255], [102, 178, 255],
                        [51, 153, 255], [255, 153, 153], [255, 102, 102],
                        [255, 51, 51], [153, 255, 153], [102, 255, 102],
                        [51, 255, 51], [0, 255, 0], [0, 0, 255], [255, 0, 0],
                        [255, 255, 255]])
    skeleton = [[16, 14], [14, 12], [17, 15], [15, 13], [12, 13], [6, 12],
                [7, 13], [6, 7], [6, 8], [7, 9], [8, 10], [9, 11], [2, 3],
                [1, 2], [1, 3], [2, 4], [3, 5], [4, 6], [5, 7]]

    pose_limb_color = palette[[
        0, 0, 0, 0, 7, 7, 7, 9, 9, 9, 9, 9, 16, 16, 16, 16, 16, 16, 16
    ]]
    pose_kpt_color = palette[[
        16, 16, 16, 16, 16, 9, 9, 9, 9, 9, 9, 0, 0, 0, 0, 0, 0
    ]]
    
    img = show_result(
        img,
        result,
        skeleton,
        radius=radius,
        thickness=thickness,
        pose_kpt_color=pose_kpt_color,
        pose_limb_color=pose_limb_color,
        kpt_score_thr=kpt_score_thr,
        show=show,
        out_file=out_file)
    return img

def single_img_visualise(output, image_size, img_name, out_file, udp=False,thickness=1,radius=4):
        
    cfg = define_cfg(udp)
    
    if cfg['use_udp']:
        base_size = (image_size,image_size)
        center = np.array([(image_size-1)/2.0,(image_size-1)/2.0]) 
        scale = np.array([image_size-1,image_size-1])
    else:
        base_size = (image_size,image_size)
        center = np.array([image_size/2,image_size/2]) 
        scale = np.array([image_size/200,image_size/200])
    
    parser = HeatmapParser(cfg)
    
    result = {}

    outputs_flipped = None
    
    aggregated_heatmaps, tags = get_multi_stage_outputs(
        output,
        outputs_flipped,
        cfg['num_joints'],
        cfg['with_heatmaps'],
        cfg['with_ae'],
        cfg['tag_per_joint'],
        cfg['flip_index'],
        cfg['project2image'],
        base_size,
        align_corners=cfg['use_udp'])

    grouped, scores = parser.parse(aggregated_heatmaps, tags,
        cfg['adjust'],
        cfg['refine'])

    preds = get_group_preds(
        grouped,
        center,
        scale, [aggregated_heatmaps.shape[3],
                aggregated_heatmaps.shape[2]],
        use_udp=cfg['use_udp'])
    
    actual_size = cv2.imread(img_name).shape
    k = [actual_size[1],actual_size[0]]
    final_size = image_size
    # for converting the keypoints back to the original image

    if k[1]<k[0]:
        scale_it = final_size/k[0]
        value_it = (final_size - scale_it*k[1])/2
        for i in range(len(preds)):
            for j in range(len(preds[0])):
                preds[i][j][0] = preds[i][j][0]/scale_it
                preds[i][j][1] = (preds[i][j][1]-value_it)/scale_it
    else:
        scale_it = final_size/k[1]
        value_it = (final_size - scale_it*k[0])/2
        for i in range(len(preds)):
            for j in range(len(preds[0])):
                preds[i][j][1] = preds[i][j][1]/scale_it
                preds[i][j][0] = (preds[i][j][0]-value_it)/scale_it
                
    image_paths = []
    image_paths.append(img_name)

    output_heatmap = None

    result['preds'] = preds
    result['scores'] = scores
    result['image_paths'] = img_name
    result['output_heatmap'] = output_heatmap
    
    pose_results = []
    for idx, pred in enumerate(result['preds']):
        area = (np.max(pred[:, 0]) - np.min(pred[:, 0])) * (
            np.max(pred[:, 1]) - np.min(pred[:, 1]))
        pose_results.append({
            'keypoints': pred[:, :3],
            'score': result['scores'][idx],
            'area': area,
        })

    pose_nms_thr=0.9
    keep = oks_nms(pose_results, pose_nms_thr, sigmas=None)
    pose_results = [pose_results[_keep] for _keep in keep]
    
    output_image = vis_pose_result(
        img_name,
        pose_results,
        kpt_score_thr=0.3,
        show=False,
        out_file=out_file,
        thickness=thickness,
        radius=radius)
    
    return output_image
                
    
