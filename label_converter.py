# Resizes labels
import argparse
import glob
import json
import os
from pathlib import Path

"""
This script converts labels located inside a Tarsier Label folder based on a resize transform.
As RGB images are resized to have 512 pixel height, labels have to be tranformed accordingly to be used for EagleNet training.
"""

# Example call: python label_converter.py path/to/Labels scale_height 512
parser = argparse.ArgumentParser(description='Resize labels folder.')
parser.add_argument('parent_dir', type=str, help='Labels parent dir')
parser.add_argument('preprocess', type=str, default='scale_height', help='How to process labels')
parser.add_argument('value', type=int, default=512, help='Preprocess value')
args = parser.parse_args()

# Tarsier json format
# json -> 'img_id1', 'img_id2' -> 'labels' -> list(dict)
#                              -> 'depth', ..., 'width', 'height'

# Find all label files
files = glob.glob(args.parent_dir + '/**/*.json', recursive=True)
# Iterate over all of them
for file in files:
    # Open label file
    j = json.load(open(file))
    # Get all image ids for that label file
    ids = j.keys()
    # Iterate over all image ids
    for id in ids:
        # Get original width/height
        width = j[id]['width']
        height = j[id]['height']
        # Calculate scaling value
        if args.preprocess == 'scale_height':
            s = args.value / height
        elif args.preprocess == 'scale_width':
            s = args.value / width
        else:
            raise Exception('Label preprocessing not implemented...')

        # Calculate new width/height
        width_new = int(round(width * s))
        height_new = int(round(height * s))

        # Resize labels
        for label in j[id]['labels']:
            label['bbox'][0] = int(round(label['bbox'][0] * s))
            label['bbox'][1] = int(round(label['bbox'][1] * s))
            label['bbox'][2] = int(round(label['bbox'][2] * s))
            label['bbox'][3] = int(round(label['bbox'][3] * s))

        # Set new width, height
        j[id]['width'] = width_new
        j[id]['height'] = height_new

    # Recreate directory structure inside 'lc_output' folder
    target_dir = os.path.join('lc_output', os.path.dirname(file))
    basename = os.path.basename(file)
    Path(target_dir).mkdir(exist_ok=True, parents=True)
    # Export resized label json file
    with open(os.path.join(target_dir, basename), 'w') as f:
        json.dump(j, f)
