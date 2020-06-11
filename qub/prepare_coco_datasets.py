from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
from __future__ import unicode_literals

import argparse
import os
import sys

from pathlib import Path
from pycocotools import mask as mask_util
import imageio
import json
import numpy as np
import scipy.ndimage


from mask_encoding import encode_rle, parse_segments_from_outlines, parse_xml_annotations, \
    regions_to_rle
from nuclei_utils import rescale_0_255, image_ids_in



ROOT_DIR = Path('C:/Users/3055638/Documents/code/detectron2/dataset')
DATASET_WORKING_DIR = ROOT_DIR / 'patch-512'


def load_images(raw_train_images_dir, raw_train_annotations_dir, ids, dataset):
    images = []
    rles_list = []
    image_sizes = []
    im_ids = []
    for id in ids:
        im_path = str(raw_train_images_dir / id)
        image = imageio.imread(im_path)

        if len(image.shape) == 2:
            image = np.stack([image, image, image]).transpose((1, 2, 0))

        image = image[:, :, :3]  # remove the alpha channel as it is not used

        if dataset == 'cd3':
            outline_path = (raw_train_annotations_dir / (id.split('.')[0] + '.png')).as_posix()

        rles = parse_segments_from_outlines(outline_path, dataset)
        if not rles:
            continue

        rles_list.append(rles)
        images.append(image)
        image_sizes.append(image.shape[:2])
        im_ids.append(id)

    return images, rles_list, image_sizes, im_ids


def tile_image(I, sz=512, resize=None, order=3):
    print(I.shape)
    height, width, _= I.shape
    import scipy.ndimage

    chunks = []
    names = []
    for h in range(0, height, sz):
        for w in range(0, width, sz):
            w_end = w + sz
            h_end = h + sz
            c = I[w:w_end, h:h_end]
            n = '{}_{}_x_{}_{}'.format(w, w_end, h, h_end)
            if resize:
                c = scipy.ndimage.zoom(c, (resize / float(sz), resize / float(sz), 1), order=order)
            chunks.append(c)
            names.append(n)

    return chunks, names


def get_all_tiles(I, sizes, resize, order=3):
    tiles = []
    names = []

    for sz in sizes:  # [128, 256, 512, 1000]:
        c, n = tile_image(I, sz, resize, order=order)
        tiles.extend(c)
        names.extend(n)
        print('chunk created')
    return tiles, names


def filter_masks(M):
    masks = []
    for idx in range(M.shape[2]):  # for each mask channel
        if M[:, :, idx].sum() < 5:
            continue
        masks.append(M[:, :, idx])
    if masks:
        return True, np.stack(masks).transpose((1, 2, 0))  # put channels back to place
    return False, None


def convert_union_mask_to_masks(mask_union):
    from skimage import measure
    assert mask_union.shape[2] == 1

    blobs_labels = measure.label(mask_union[:, :, 0], background=0)
    masks = []
    for idx in range(1, blobs_labels.max() + 1):  # for each mask channel
        masks.append(blobs_labels == idx)

    return np.stack(masks).transpose((1, 2, 0))  # put channels back to place


def preprocess_as_tiles(orig_images, orig_rles_list, orig_im_ids):
    images = []
    rles_list = []
    image_sizes = []
    image_names = []

    for I, rles, im_name in zip(orig_images, orig_rles_list, orig_im_ids):
        M = mask_util.decode(rles)

        mtiles, _ = get_all_tiles(M, [512], 512, order=1)
        tiles, names = get_all_tiles(I, [512], 512, order=3)

        for t, m, n in zip(tiles, mtiles, names):
            success, m = filter_masks(m)
            if success:
                rles_list.append(mask_util.encode(np.asarray(m, order='F')))
                images.append(t)
                image_sizes.append(t.shape[:2])
                image_names.append('{}:{}'.format(im_name, n))
                print('Image')
            else:
                print('Failed Image')
    return images, rles_list, image_sizes, image_names


def get_image_data(image, rles, image_id, image_filename, size, class_name, train_image_dir):
    im_metadata = {
        'file_name': image_filename + '.png',
        'height': size[0],
        'id': image_id,
        'width': size[1],
        'nuclei_class': class_name,
        # 'is_grey_scale': is_grey_scale_mat(image)
    }

    annotations = []
    global annotation_id
    for rle in rles:
        encoded_segment = encode_rle(rle, annotation_id, image_id)
        if encoded_segment['area'] > 0:
            annotations.append(encoded_segment)
            annotation_id += 1
        else:
            from pprint import pprint as pp
            pp(encoded_segment)

    if annotations:
        file_name = train_image_dir / (image_filename + '.png')
        imageio.imsave(file_name.as_posix(), image)

    return im_metadata, annotations


annotation_id = 0


def prepare_cd3():
    dataset_name = 'cd3'

    raw_input_dir = ROOT_DIR / 'patch-512'

    raw_train_images_dir = raw_input_dir / 'images/validation'
    raw_train_annotations_dir = raw_input_dir / 'annotations/validation'

    train_images_output_dir = DATASET_WORKING_DIR / dataset_name

    try:
        os.mkdir(train_images_output_dir.as_posix())
    except:
        pass

    annotations_file_path = DATASET_WORKING_DIR / 'cd3_val.json'

    im_ids = image_ids_in(raw_train_images_dir)
    images, rle_lists, image_sizes, train_image_ids = load_images(raw_train_images_dir,
                                                                  raw_train_annotations_dir,
                                                                  im_ids,
                                                                  dataset_name)

    im_count = 0
    dataset_structure = {
        'annotations': [],
        'categories': [
            {
                'id': 1,
                'name': 'cd3'
            }
        ],
        'images': [],
        'info': {
            'description': 'coco format cd3 Dataset',
        }
    }

    for im, im_id, masks, sz in zip(images, train_image_ids, rle_lists, image_sizes):
        im_metadata, annotations_res = get_image_data(im, masks, im_count, im_id, sz, 'cd3',
                                                      train_images_output_dir)
        dataset_structure['images'].append(im_metadata)
        dataset_structure['annotations'].extend(annotations_res)
        im_count += 1

    json.dump(dataset_structure, open(annotations_file_path.as_posix(), 'w'))


def parse_args():
    parser = argparse.ArgumentParser(description='Prepare external datasets')
    parser.add_argument(
        '--root-data-dir',
        dest='root_data_dir',
        help='Path to the root data dir',
        default=None,
        type=str
    )

    if len(sys.argv) == 1:
        parser.print_help()
        sys.exit(1)
    return parser.parse_args()


def main():
    prepare_cd3()


if __name__ == '__main__':
    args = parse_args()
    ROOT_DIR = Path(args.root_data_dir)
    main()
