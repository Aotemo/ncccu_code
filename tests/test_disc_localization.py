"""Independent image geometry oracles; no task labels or model forward."""
import unittest
from PIL import Image,ImageDraw,ImageOps
from retrain.preprocessing import locate_optic_disc


def fundus_fixture():
    image=Image.new('RGB',(480,320),(0,0,0));draw=ImageDraw.Draw(image)
    draw.ellipse((8,8,472,312),fill=(110,75,55))
    draw.ellipse((106,136,154,184),fill=(210,185,130))
    # A brighter, larger avascular object challenges brightness-only selection.
    draw.ellipse((315,75,385,145),fill=(255,250,230))
    paths=[[(130,160),(90,125),(45,100)],[(130,160),(95,185),(50,220)],
           [(130,160),(160,110),(235,55)],[(130,160),(170,205),(270,265)],
           [(130,160),(190,155),(265,135)],[(130,160),(115,220),(110,275)]]
    for points in paths:draw.line(points,fill=(45,20,15),width=5)
    return image


class DiscLocalizationTest(unittest.TestCase):
    def test_constant_image_is_unknown_without_brightness_fallback(self):
        image=Image.new('RGB',(480,320),(130,130,130));before=image.tobytes()
        result=locate_optic_disc(image)
        self.assertEqual(result['status'],'unknown')
        self.assertIsNone(result['center_xy'])
        self.assertEqual(before,image.tobytes())

    def test_vascular_disc_wins_over_brighter_avascular_circle(self):
        image=fundus_fixture();before=image.tobytes();result=locate_optic_disc(image)
        self.assertEqual(result['status'],'located',result)
        x,y=result['center_xy']
        self.assertLessEqual((x-130)**2+(y-160)**2,12**2)
        self.assertEqual(before,image.tobytes())
        box=result['native_box'];self.assertEqual(box[2]-box[0],160)
        self.assertLessEqual(box[0],82);self.assertGreaterEqual(box[2],178)
        self.assertLessEqual(box[1],112);self.assertGreaterEqual(box[3],208)

    def test_flip_and_quarter_turn_preserve_pixel_coordinates(self):
        image=fundus_fixture();point=locate_optic_disc(image)['center_xy']
        transforms=[(ImageOps.mirror(image),(479-point[0],point[1])),
                    (image.transpose(Image.Transpose.ROTATE_90),(point[1],479-point[0]))]
        for transformed,expected in transforms:
            result=locate_optic_disc(transformed);self.assertEqual(result['status'],'located',result)
            self.assertLessEqual(sum((a-b)**2 for a,b in zip(result['center_xy'],expected)),2**2)
