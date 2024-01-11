import os
import cv2
from pathlib import Path
from magicwand import SelectionWindow
import numpy as np
root_dir = Path(__file__).parents[0].as_posix()


finger_component_class_color = {0: ([255, 255, 255], "ridge"), # white
                                1: ([0, 255, 0], "ridge_end"), # green
                                2: ([0, 0, 255], "core"), # red
                                3: ([255, 0, 0], "delta"), # blue
                                4: ([0, 255, 255], "bifraction"), # yello
                                5: ([255, 255, 0], "ice_land and dot"), # sky blue
                                6: ([255, 0, 255], "hook and duble bifraction"), # Purple
                                }
path = os.path.join(root_dir, "data")
list_dataset = Path(path).glob("**/*.png")
mask_dir_path = os.path.join(root_dir, "mask")
os.makedirs(mask_dir_path, exist_ok=True)
for p in list_dataset:
    print("read file", p.as_posix())
    img = cv2.imread(p.as_posix())
    # img = cv2.bitwise_not(img)
    img = cv2.GaussianBlur(img, (3, 3), 0, 0)
    img = cv2.adaptiveThreshold(cv2.cvtColor(img, cv2.COLOR_BGR2GRAY), 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, 33, 0)
    # open window to segment
    img = cv2.cvtColor(img, cv2.COLOR_GRAY2RGB)
    window = SelectionWindow(img, finger_component_class_color, img_path=p.as_posix())
    window.show()
    rgb_mask = window.rgb_mask
    mask_name = "mask_" + p.name
    mask_path = os.path.join(mask_dir_path, mask_name)
    cv2.imwrite(mask_path, rgb_mask)
    print("mask file wroted on", mask_path)