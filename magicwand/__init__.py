import cv2 as cv
import numpy as np


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
    def __init__(self, img, class_color, name="Magic Wand Selector", connectivity=4, tolerance=32):
        self.org_img = img.copy()
        self.connectivity = connectivity
        self.name = name
        h, w = img.shape[:2]
        self.img = img.copy()
        self.mask = np.zeros((h, w), dtype=np.uint8)
        # ========
        self.rgb_mask = np.zeros((h, w, 3), dtype=np.uint8)
        self.class_color = class_color
        self.click_point = []
        self.cut_points = []
        self.line_mask = np.zeros((h, w), dtype=np.uint8)
        self.drawing = False
        # =======
        self._flood_mask = np.zeros((h + 2, w + 2), dtype=np.uint8)
        self._flood_fill_flags = (
            self.connectivity | cv.FLOODFILL_FIXED_RANGE | cv.FLOODFILL_MASK_ONLY | 255 << 8
        )  # 255 << 8 tells to fill with the value 255
        cv.namedWindow(self.name, cv.WINDOW_NORMAL)
        # =========
        cv.namedWindow("rgb final mask", cv.WINDOW_NORMAL)
        cv.namedWindow("binary last mask", cv.WINDOW_NORMAL)
        # =========
        self.tolerance = (tolerance,) * 3
        cv.createTrackbar(
            "Tolerance", self.name, tolerance, 255, self._trackbar_callback
        )
        cv.setMouseCallback(self.name, self._mouse_callback)
        # ======
        self.eraser_size = 1
        cv.createTrackbar(
            "Eraser Size", "rgb final mask", 1, 10, self._eraser_size_callback
        )
        cv.setMouseCallback("rgb final mask", self._mouse_rgb_callback)

    def _trackbar_callback(self, pos):
        self.tolerance = (pos,) * 3
    
    def _eraser_size_callback(self, size):
        self.eraser_size = size

    def _cut_selected_area(self, x, y):
        self.cut_points.append([x, y])
        if len(self.cut_points) % 2 == 0:
            last2cut_points = self.cut_points[-2:]
            # draw line in preview image
            cv.line(self.img, tuple(last2cut_points[0]), tuple(last2cut_points[1]), (80, 80, 80), thickness=3)
            # draw line in line_mask
            line = cv.line(self.line_mask, tuple(last2cut_points[0]), tuple(last2cut_points[1]), (255, 255, 255))
            # get places which line cut segment
            roi_line = cv.bitwise_and(self.mask, line)
            # get indexes of roi_line then calculate align of segmentation to cur (X, Y)
            index_of_cut_roi = np.where(roi_line == 255)
            min_x_index = np.min(index_of_cut_roi[1])
            min_y_index = np.min(index_of_cut_roi[0])
            max_x_index = np.max(index_of_cut_roi[1])
            max_y_index = np.max(index_of_cut_roi[0])
            align_base_on_x_y = "Y" if (max_y_index - min_y_index) > (max_x_index - min_x_index) else "X"
            # cut segment base on roi_line
            if align_base_on_x_y == "Y":
                if self.click_point[0] < max_x_index:
                    # get left. or cut right of segmentation
                    self.mask[:, max_x_index:] = 0
                    print("get left")
                else:
                    # get right. or cut left of segmentation
                    self.mask[:, :max_x_index] = 0
                    print("get right")
            else:
                if self.click_point[1] < max_y_index:
                    # get top. or cut down of segmentation
                    self.mask[max_y_index:, :] = 0
                    print("get top")
                else:
                    # get down. or cut top of segmentation
                    self.mask[:max_y_index, :] = 0
                    print("get down")
                    
            cv.imwrite("/home/mehdi/Documents/projects/Protect_the_Great_Barrier_Reef/resana/seg1.png", self.mask.astype("uint8"))
            h, w = self.img.shape[:2]
            self.line_mask = np.zeros((h, w), dtype=np.uint8)
    
    def _mouse_rgb_callback(self, event, x, y, flags, *userdata):
        
        if event == cv.EVENT_LBUTTONDOWN:
            self.drawing = True
        elif event==cv.EVENT_MOUSEMOVE:
            if self.drawing == True:
                cv.circle(self.rgb_mask, (x,y), self.eraser_size,(0,0,0),-1)
        elif event == cv.EVENT_LBUTTONUP:
            self.drawing = False
        self._update()
    
    def _mouse_callback(self, event, x, y, flags, *userdata):

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
            self.mask = cv.bitwise_and(self.mask, cv.bitwise_not(flood_mask))
            self.click_point = [x, y]
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
        self.show_image(viz)
        cv.displayStatusBar(self.name, ", ".join((meanstr, stdstr)))

    def show_image(self, viz):
        cv.imshow(self.name, viz)
        cv.imshow("binary last mask", self.mask)
        cv.imshow("rgb final mask", self.rgb_mask)
        
    def destroyWindows(self):
        cv.destroyWindow(self.name)
        cv.destroyWindow("rgb final mask")
        cv.destroyWindow("binary last mask")
    
    def set_segment_class(self, key):
        b_id, g_id, r_id = self.class_color[int(chr(key))]
        self.rgb_mask[self.mask == 255, 0] = b_id
        self.rgb_mask[self.mask == 255, 1] = g_id
        self.rgb_mask[self.mask == 255, 2] = r_id
        self._update()
    
    def show(self):
        """Draws a window with the supplied image."""
        self._update()
        print("Press [q] or [esc] to close the window and Press [c] to reset windows")
        while True:
            k = cv.waitKey() & 0xFF
            print(k)
            if k in (ord("q"), ord("\x1b")):
                self.destroyWindows()
                break
            if k in [ord(str(k)) for k in self.class_color.keys()]:
                self.set_segment_class(k)
            if k == ord("\x08"): # back space
                pass
            if k == ord("c"):
                self._reset_window()
    
    def _reset_window(self):
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
        