import numpy as np
import os
from PIL import Image
import json
import cv2

IMG_EXTENSIONS = [
    '.jpg', '.JPG', '.jpeg', '.JPEG',
    '.png', '.PNG', '.ppm', '.PPM', '.bmp', '.BMP',
    '.tif', '.TIF', '.tiff', '.TIFF',
]


def is_image_file(filename):
    return any(filename.endswith(extension) for extension in IMG_EXTENSIONS)


def make_dataset(data_path, max_dataset_size=float("inf"),
                 return_mask=False,
                 query_str=None):
    # wulin 2021.5.26
    if query_str is None:
        query_str = ['low', 'normal', 'mask']
    query_len = len(query_str)-1  # last for mask
    mask_paths = [[]]
    image_paths = [[] for _ in range(query_len)]
    assert os.path.isdir(data_path), '%s is not a valid directory' % data_path
    for root, _, fnames in sorted(os.walk(data_path)):
        fnames = sorted(fnames)
        invariant_fname = []
        for fname in fnames:
            if is_image_file(fname):
                fname_split = fname.split("_") # llhdr_20210226141843_0ev_2944x2208_2944x2208
                fname_pre_suffix = (tuple(fname_split[:2]), tuple(fname_split[3:]))
                if fname_pre_suffix not in invariant_fname:
                    invariant_fname.append(fname_pre_suffix)
        for item in invariant_fname:
            for query_idx in range(query_len):
                image_path = os.path.join(
                    root, "_".join(list(item[0]) + [query_str[query_idx]] + list(item[1])))
                assert os.path.isfile(image_path)
                image_paths[query_idx].append(image_path)
            if return_mask:
                # -------------------
                mask_path = os.path.join(
                    os.path.abspath(os.path.join(root, '..')), 'mask',
                    "_".join(list(item[0]) + [query_str[-1]] + list(item[1])))
                mask_path = mask_path.replace('.jpg', '.json')
                # -------------------
                assert os.path.isfile(mask_path)
                mask_paths[0].append(mask_path)
            else:
                mask_paths[0].append(None)

    max_len = min(max_dataset_size, len(image_paths[0]))
    assert len(mask_paths[0]) == len(image_paths[0])
    for query_idx in range(query_len):
        image_paths[query_idx] = image_paths[query_idx][:max_len]
    mask_paths[0] = mask_paths[0][:max_len]
    return image_paths, mask_paths


def extract_mask_from_json(mask_path):
    """
    inside is 255, outside is 0
    # {'xmin': 406, 'ymin': 158, 'xmax': 528, 'ymax': 250}
    --------> (xmin, xmax)
    |
    v (ymin, ymax)
    """
    with open(mask_path, 'r') as load_f:
        load_dict = json.load(load_f)
        width, height = load_dict['size']['width'], load_dict['size']['height']
        mask_dict = load_dict['outputs']['object'][0]['bndbox']
    min_x, min_y, max_x, max_y = \
        mask_dict['xmin'], mask_dict['ymin'], mask_dict['xmax'], mask_dict['ymax']
    mask = np.zeros([height, width]).astype('uint8')
    mask = cv2.rectangle(mask, (min_x, min_y), (max_x, max_y), 255, thickness=-1)
    return mask


if __name__ == '__main__':
    """
    if you want low-normal data
    elif you want low-mid-mormal data
    """
    # data_path = "/mnt/19sdc/data/ARC_LLHDR_DUMP_LSI1/resize_0m2m3ev/train"
    # image_paths, mask_paths = make_dataset(data_path, max_dataset_size=float("inf"),
    #                                     return_mask=False,
    #                                     query_str=['-3ev', '-2ev', '0ev', 'mask'])

    data_path = "/mnt/19sdc/data/DB_1015_1103_1118/resize_lownormal/train"
    image_paths, mask_paths = make_dataset(data_path, max_dataset_size=float("inf"),
                                 return_mask=True,
                                 query_str=['low', 'normal', 'mask'])



