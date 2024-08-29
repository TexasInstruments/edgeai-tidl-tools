from edgeai_benchmark.utils import AttrDict

def create_model_config(
        source=AttrDict(),
        preprocess=AttrDict(),
        session=AttrDict(),
        postprocess=AttrDict(),
        extra_info=AttrDict(),
        task_type = None
        ):
    pipeline_config = AttrDict(
        source = source,
        preprocess=preprocess,
        session=session,
        postprocess=postprocess,
        task_type = task_type,
        metric=AttrDict(),
        extra_info = extra_info
        )
    return pipeline_config

def coco_det_label_offset_80to90(label_offset=1):
    coco_label_table = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 13, 14, 15, 16, 17, 18, 19, 20,
                         21, 22, 23, 24, 25, 27, 28, 31, 32, 33, 34, 35, 36, 37, 38, 39, 40,
                         41, 42, 43, 44, 46, 47, 48, 49, 50, 51, 52, 53, 54, 55, 56, 57, 58, 59, 60,
                         61, 62, 63, 64, 65, 67, 70, 72, 73, 74, 75, 76, 77, 78, 79, 80,
                         81, 82, 84, 85, 86, 87, 88, 89, 90]

    if label_offset == 1:
        # 0 => 1, 1 => 2, .. 79 -> 90, 80 => 91
        coco_label_offset = {k:v for k,v in enumerate(coco_label_table)}
        coco_label_offset.update({80:91})
    elif label_offset == 0:
        # 0 => 0, 1 => 1, .. 80 => 90
        coco_label_offset = {(k+1):v for k,v in enumerate(coco_label_table)}
        coco_label_offset.update({0:0})
    else:
        assert False, f'unsupported value for label_offset {label_offset}'
    return coco_label_offset

# convert from 90 class index (typical output of a tensorflow detector) to 90 or 91 class
# (original labels of coco starts from 1, and 0 is background)
def coco_det_label_offset_90to90(label_offset=1):
    coco_label_table = range(1,91)
    if label_offset == 1:
        # 0 => 1, 1 => 2, .. 90 => 91
        coco_label_offset = {k:v for k,v in enumerate(coco_label_table)}
        coco_label_offset.update({-1:0,90:91})
    elif label_offset == 0:
        # 0 => 0, 1 => 1, .. 90 => 90
        coco_label_offset = {(k+1):v for k,v in enumerate(coco_label_table)}
        coco_label_offset.update({-1:-1,0:0})
    else:
        assert False, f'unsupported value for label_offset {label_offset}'
    return coco_label_offset

def postprocess_function(model_config, **kwargs):
    return kwargs
