import cv2 as cv
import numpy as np
from pascal_voc_writer import Writer


SHIFT_KEY = cv.EVENT_FLAG_SHIFTKEY
ALT_KEY = cv.EVENT_FLAG_ALTKEY
CTRLKEY = cv.EVENT_FLAG_CTRLKEY


def _find_exterior_contours(img):
    ret = cv.findContours(img, cv.RETR_EXTERNAL, cv.CHAIN_APPROX_SIMPLE)
    if len(ret) == 2:
        return ret[0]
    elif len(ret) == 3:
        return ret[1]
    raise Exception("Check the signature for `cv.findContours()`.")


class SelectionWindow:
    def __init__(self, img: np.uint8,
                 assistance_image:np.uint8,
                 class_color: dict, 
                 img_path:str = "",
                 name: str="Magic Wand Selector", 
                 connectivity:int=4, 
                 tolerance:int=32):
        """_summary_

        Args:
            img (np.uint8): original image
            class_color (dict): label class dict to map each key to a color exp: {0: [0,255,0], 1:[255,0,0],...}
            name (str, optional): name of original image window. Defaults to "Magic Wand Selector".
            connectivity (int, optional): _description_. Defaults to 4.
            tolerance (int, optional): tolerance of selecting area. Defaults to 32.
        """
        self.org_img = img.copy()
        self.connectivity = connectivity
        self.name = name
        h, w = img.shape[:2]
        self.img = img.copy()
        self.mask = np.zeros((h, w), dtype=np.uint8)
        # ========
        self.assistance_image = assistance_image
        self.box_size = 16
        self.rgb_mask = np.zeros((h, w, 3), dtype=np.uint8)
        self.class_color = class_color
        self.click_point = []
        self.cut_points = []
        self.orientaion_points = []
        self.line_mask = np.zeros((h, w), dtype=np.uint8)
        self.drawing = False
        self.img_path = img_path
        self.writer = Writer(img_path, w, h)
        self.last_class = -1
        # =======
        self._flood_mask = np.zeros((h + 2, w + 2), dtype=np.uint8)
        self._flood_fill_flags = (
            self.connectivity | cv.FLOODFILL_FIXED_RANGE | cv.FLOODFILL_MASK_ONLY | 255 << 8
        )  # 255 << 8 tells to fill with the value 255
        cv.namedWindow(self.name, cv.WINDOW_NORMAL)
        # =========
        
        cv.namedWindow("assistance_image", cv.WINDOW_NORMAL)
        cv.namedWindow("rgb final mask", cv.WINDOW_NORMAL)
        cv.namedWindow("binary last mask", cv.WINDOW_NORMAL)
        # =========
        self.tolerance = (tolerance,) * 3
        cv.createTrackbar(
            "Tolerance", self.name, tolerance, 255, self._trackbar_callback
        )
        cv.createTrackbar(
            "box_size", self.name, tolerance, 256, self._box_size_callback
        )
        cv.setMouseCallback(self.name, self._mouse_callback)
        # ======
        self.eraser_size = 1
        cv.createTrackbar(
            "Eraser Size", "rgb final mask", 1, 10, self._eraser_size_callback
        )
        cv.setMouseCallback("rgb final mask", self._mouse_rgb_callback)
        cv.createTrackbar(
            "Eraser Size", "binary last mask", 1, 10, self._eraser_size_callback
        )
        cv.setMouseCallback("binary last mask", self._mouse_rgb_callback)

    def _trackbar_callback(self, pos):
        """_summary_

        Args:
            pos (_type_): set position of trace bar to tolerance
        """
        self.tolerance = (pos,) * 3
    
    def _box_size_callback(self, pos):
        """_summary_

        Args:
            pos (_type_): set position of trace bar to tolerance
        """
        self.box_size = pos
        
    def _eraser_size_callback(self, size):
        """_summary_

        Args:
            size (_type_): eraser size can be set by trace bar of 'rgb final mask' window
        """
        self.eraser_size = size

    def _refresh_mask(self):
        """this func erase other area of cut line
        """
        contours, _ = cv.findContours(self.mask, cv.RETR_EXTERNAL, cv.CHAIN_APPROX_NONE)
        tmp_mask = np.zeros_like(self.mask)
        for idx, cnt in enumerate(contours):
            cv.drawContours(tmp_mask, [cnt], -1, (255, 255, 255), thickness=cv.FILLED)
            index_of_cnt = np.where(tmp_mask == 255)
            if tuple(self.click_point) not in list(zip(index_of_cnt[1].tolist(), index_of_cnt[0].tolist())):
                cv.drawContours(self.mask, [cnt], -1, (0, 0, 0), thickness=cv.FILLED)
            tmp_mask = np.zeros_like(self.mask)
                    
    def _cut_selected_area(self, x:int, y:int, orientation_flag=False):
        """get coordinates of click points. then cut the other side of the selected area

        Args:
            x (int): x coordinate
            y (int): y coordinate
        """
        if orientation_flag:
            self.orientaion_points.append([x, y])
            if len(self.orientaion_points) % 2 == 0:
                last2cut_points = self.orientaion_points[-2:]
                # draw line in preview image
                cv.arrowedLine(self.img, tuple(last2cut_points[0]), tuple(last2cut_points[1]), (200, 0, 0), thickness=3)
                self.writer.addObject(self.last_class, 
                            last2cut_points[0][0],
                            last2cut_points[0][1],
                            last2cut_points[1][0],
                            last2cut_points[1][1])
                
        else:
            self.cut_points.append([x, y])
            last2cut_points = self.cut_points[-1]
            # draw line in preview image
            x_min = last2cut_points[0] - (self.box_size // 2)
            x_max = last2cut_points[0] + (self.box_size // 2)
            y_min = last2cut_points[1] - (self.box_size // 2)
            y_max = last2cut_points[1] + (self.box_size // 2)
            
            cv.rectangle(self.img, (x_min, y_min), (x_max, y_max), (80, 80, 80), thickness=3)
            # cv.line(self.img, tuple(last2cut_points[0]), tuple(last2cut_points[1]), (80, 80, 80), thickness=3)
            # draw line in line_mask
            box = cv.rectangle(self.line_mask, (x_min, y_min), (x_max, y_max), (255, 255, 255), thickness=-1)
            # line = cv.line(self.line_mask, tuple(last2cut_points[0]), tuple(last2cut_points[1]), (255, 255, 255), thickness=3)
            # get places which line cut segment
            roi_box = cv.bitwise_and(self.mask, box)
            self.mask = roi_box.copy()
            # get indexes of roi_line then calculate align of segmentation to cur (X, Y)
            # index_of_cut_roi = np.where(roi_line == 255)
            # # cut segment base on roi_line           
            # for x, y in zip(index_of_cut_roi[1], index_of_cut_roi[0]):
            #     self.mask[y, x] = 0
            self._refresh_mask()
            h, w = self.img.shape[:2]
            self.line_mask = np.zeros((h, w), dtype=np.uint8)        
    
    def _mouse_rgb_callback(self, event, x:int, y:int, flags, *userdata):
        """this is an eraser callback

        Args:
            event (_type_): cv EVENT_LBUTTONDOWN
            x (_type_): x coordinate
            y (_type_): y coordinate
            flags (_type_): None
        """
        if event == cv.EVENT_LBUTTONDOWN:
            self.drawing = True
        elif event==cv.EVENT_MOUSEMOVE:
            if self.drawing == True:
                cv.circle(self.rgb_mask, (x,y), self.eraser_size,(0,0,0),-1)
                cv.circle(self.mask, (x,y), self.eraser_size,(0,0,0),-1)
        elif event == cv.EVENT_LBUTTONUP:
            self.drawing = False
        self._update()
    
    def _mouse_callback(self, event, x:int, y:int, flags, *userdata):
        """this is original image window callback

        Args:
            event (_type_): cv EVENT_LBUTTONDOWN
            x (int): x coordinate
            y (int): y coordinate
            flags (_type_): None
        """

        if event != cv.EVENT_LBUTTONDOWN:
            return
        modifier = flags & (ALT_KEY + SHIFT_KEY + CTRLKEY)
        self._flood_mask[:] = 0
        cv.floodFill(
            self.img,
            self._flood_mask,
            (x, y),
            0,
            self.tolerance,
            self.tolerance,
            self._flood_fill_flags,
        )
        flood_mask = self._flood_mask[1:-1, 1:-1].copy()

        if modifier == (ALT_KEY + SHIFT_KEY):
            self.mask = cv.bitwise_and(self.mask, flood_mask)
            self.click_point = [x, y]
        elif modifier == SHIFT_KEY:
            self.mask = cv.bitwise_or(self.mask, flood_mask)
            self.click_point = [x, y]
        elif modifier == ALT_KEY:
            self.mask = cv.threshold(cv.cvtColor(self.img, cv.COLOR_RGB2GRAY), 200, 255, cv.THRESH_BINARY_INV)[1]
            self.mask = cv.bitwise_not(self.mask)
            # self.mask = cv.bitwise_and(self.mask, cv.bitwise_not(flood_mask))
        elif modifier == (ALT_KEY + CTRLKEY):
            self._cut_selected_area(x, y, orientation_flag=True)
        elif modifier == CTRLKEY:
            self._cut_selected_area(x, y)
        else:
            self.mask = flood_mask
            self.click_point = [x, y]
        self._update()

        

    def _update(self):
        """Updates an image in the already drawn window."""
        viz = self.img.copy()
        contours = _find_exterior_contours(self.mask)
        viz = cv.drawContours(viz, contours, -1, color=(80,) * 3, thickness=-1)
        viz = cv.addWeighted(self.img, 0.75, viz, 0.25, 0)
        viz = cv.drawContours(viz, contours, -1, color=(80,) * 3, thickness=3)

        self.mean, self.stddev = cv.meanStdDev(self.img, mask=self.mask)
        meanstr = "mean=({:.2f}, {:.2f}, {:.2f})".format(*self.mean[:, 0])
        stdstr = "std=({:.2f}, {:.2f}, {:.2f})".format(*self.stddev[:, 0])
        self._show_image(viz)
        cv.displayStatusBar(self.name, ", ".join((meanstr, stdstr)))

    
    def _show_image(self, viz:np.uint8):
        """show all results windows

        Args:
            viz (np.uint8): feedbacked original image
        """
        cv.imshow(self.name, viz)
        cv.imshow("assistance_image", self.assistance_image)
        cv.imshow("binary last mask", self.mask)
        cv.imshow("rgb final mask", self.rgb_mask)
        
    def _destroyWindows(self):
        """destroy all opened windows
        """
        cv.destroyWindow(self.name)
    
        cv.destroyWindow("assistance_image")
        cv.destroyWindow("rgb final mask")
        cv.destroyWindow("binary last mask")
    
    def _set_segment_class(self, key):
        """Colors segment of selected area

        Args:
            key (key of 'class_color'): _description_
        """
        key_id = int(chr(key))
        self.last_class = self.class_color[key_id][1]
        b_id, g_id, r_id = self.class_color[key_id][0]
        self.rgb_mask[self.mask == 255, 0] = b_id
        self.rgb_mask[self.mask == 255, 1] = g_id
        self.rgb_mask[self.mask == 255, 2] = r_id
        self._update()
    
    def pascal_voc_save(self):
        file_name = self.img_path.split("/")[-1]
        file_extention = file_name.split(".")[-1]
        self.writer.save(self.img_path.replace(file_extention, "xml"))
    
    def show(self):
        """Draws a window with the supplied image."""
        self._update()
        print("Press [q] or [esc] to close the window and Press [c] to reset windows")
        while True:
            k = cv.waitKey() & 0xFF
            if k in (ord("q"), ord("\x1b")):
                self.pascal_voc_save()
                self._destroyWindows()
                break
            if k in [ord(str(k)) for k in self.class_color.keys()]:
                self._set_segment_class(k)
            if k == ord("\x08"): # back space
                pass
            if k == ord("c"):
                self._reset_window()
    
    def _reset_window(self):
        """with pressing [c] button. reset all selected areas
        """
        self.img = self.org_img.copy()
        h, w = self.img.shape[:2]
        self.mask = np.zeros((h, w), dtype=np.uint8)
        self._flood_mask = np.zeros((h + 2, w + 2), dtype=np.uint8)
        self._flood_fill_flags = (
            self.connectivity | cv.FLOODFILL_FIXED_RANGE | cv.FLOODFILL_MASK_ONLY | 255 << 8
        )  # 255 << 8 tells to fill with the value 255
        self.line_mask = np.zeros((h, w), dtype=np.uint8)
        self.click_point = []
        self.cut_points = []
        self.drawing = False
        self._update()
        