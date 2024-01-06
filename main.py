import os
import cv2
from pathlib import Path
from magicwand import SelectionWindow

root_dir = Path(__file__).parents[0].as_posix()

finger_component_class = {0: "ridge",
                          1: "ridge_end", 
                          2: "core", 
                          3: "delta", 
                          4: "bifraction", 
                          5: "ice_land and dot",
                          6: "hook and duble bifraction"}
finger_component_class_color = {0: [255, 255, 255], # white
                                1: [0, 255, 0], # green
                                2: [0, 0, 255], # red
                                3: [255, 0, 0], # blue
                                4: [0, 255, 255], # yello
                                5: [255, 255, 0], # sky blue
                                6: [255, 0, 255], # Purple
                                }
path = os.path.join(root_dir, "data")
list_dataset = Path(path).glob("**/*.png")
mask_dir_path = os.path.join(root_dir, "mask")
os.makedirs(mask_dir_path, exist_ok=True)
for p in list_dataset:
    print("read file", p.as_posix())
    img = cv2.imread(p.as_posix())
    # open window to segment
    window = SelectionWindow(img, finger_component_class_color)
    window.show()
    rgb_mask = window.rgb_mask
    mask_name = "mask_" + p.name
    mask_path = os.path.join(mask_dir_path, mask_name)
    cv2.imwrite(mask_path, rgb_mask)
    print("mask file wroted on", mask_path)