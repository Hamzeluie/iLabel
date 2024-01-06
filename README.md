## iLabel

Segmentation mask creator tools for labeling fingerprint images.for trainig segmentation model to featch minutiae points

![Example Image](readme-example.png)

## Getting Started

Install into a Python virtual environment, as you would any other Python project.

```sh
$ python3 -m venv venv
$ source venv/bin/activate
(venv) $ pip install git@github.com:Hamzeluie/iLabel.git
```
it has three windows as you can see in the image

# Magic Wand Selector window:
this shows the original image. It gives you the ability to select an area like a magic wand tool. also, you can add separated area with "SHIFT_KEY + left button click", deselect the area with "ALT_KEY + left button click" or **you can custom cut a line shape selected area by pressing CTRL + left button click to the start point of cut line then again CTRL + left button click to the endpoint to cut a line selected area.**
 
# binary last mask  window: 
this window shows the current selected area like a binary mask to allow you to manage segments better.

# rgb final mask  window:
if you want to multi-segment an image you should pass to the "SelectionWindow" class a  parameter "class_color". then when you select the interest area to classify the segment you can set a label color to the segment by pressing the key value of your passed parameter "class_color" and you can see the results in this window. also, you can erase the segment area by holding the left button click

