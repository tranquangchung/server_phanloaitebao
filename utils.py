"""
Mask R-CNN
Common utility functions and classes.

Copyright (c) 2017 Matterport, Inc.
Licensed under the MIT License (see LICENSE for details)
Written by Waleed Abdulla
"""

import sys
import os
import math
import random

def remove_files(all_file, file_name):
    has_file_delete = False
    for f in all_file:
        if file_name in f:
            ## delete file here
            has_file_delete = True
            os.remove(f)
    return has_file_delete

def get_list_files(label="all"):
    if label=="all":
        files = glob.glob('./static/data_labeling/*/*')
    else:
        files = glob.glob('./static/data_labeling/{0}/*'.format(label))
    return sorted(files, key=os.path.getmtime)

