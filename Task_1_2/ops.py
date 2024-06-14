import numpy as np
import skimage as sk
from PIL import Image, ImageOps, ImageDraw
import math
from io import BytesIO
import cv2
from pkg_resources import resource_filename
from wand.image import Image as WandImage
import random
import torch
from torchvision.transforms import *
from tacobox import Taco


class SaltNoise:
    """
    Salt noise operation. Adapted from https://github.com/roatienza/straug/blob/main/straug/noise.py
    """
    def __init__(self, rng=None):
        self.rng = np.random.default_rng() if rng is None else rng

    def __call__(self, img, mag=-1, prob=1.):
        if self.rng.uniform(0, 1) > prob:
            return img

        b = [.03, .07, .11]
        if mag < 0 or mag >= len(b):
            index = 0
        else:
            index = mag
        a = b[index]
        c = self.rng.uniform(a, a + .04)
        img = sk.util.random_noise(np.asarray(img) / 255., mode='salt', amount=c) * 255
        return Image.fromarray(img.astype(np.uint8))


class Snow:
    """
    Snow operation. Taken from https://github.com/roatienza/straug/blob/main/straug/weather.py
    """
    def __init__(self, rng=None):
        self.rng = np.random.default_rng() if rng is None else rng

    def __call__(self, img, mag=-1, prob=1.):
        if self.rng.uniform(0, 1) > prob:
            return img

        w, h = img.size
        c = [(0.1, 0.3, 3, 0.5, 10, 4, 0.8),
             (0.2, 0.3, 2, 0.5, 12, 4, 0.7),
             (0.55, 0.3, 4, 0.9, 12, 8, 0.7)]
        if mag < 0 or mag >= len(c):
            index = self.rng.integers(0, len(c))
        else:
            index = mag
        c = c[index]

        n_channels = len(img.getbands())
        isgray = n_channels == 1

        img = np.asarray(img, dtype=np.float32) / 255.
        if isgray:
            img = np.expand_dims(img, axis=2)
            img = np.repeat(img, 3, axis=2)

        snow_layer = self.rng.normal(size=img.shape[:2], loc=c[0], scale=c[1])

        snow_layer[snow_layer < c[3]] = 0

        snow_layer = Image.fromarray((np.clip(snow_layer.squeeze(), 0, 1) * 255).astype(np.uint8), mode='L')
        output = BytesIO()
        snow_layer.save(output, format='PNG')
        snow_layer = WandImage(blob=output.getvalue())

        snow_layer.motion_blur(radius=c[4], sigma=c[5], angle=self.rng.uniform(-135, -45))

        snow_layer = cv2.imdecode(np.frombuffer(snow_layer.make_blob(), np.uint8),
                                  cv2.IMREAD_UNCHANGED) / 255.

        snow_layer = snow_layer[..., np.newaxis]

        img = c[6] * img
        gray_img = (1 - c[6]) * np.maximum(img, cv2.cvtColor(img, cv2.COLOR_RGB2GRAY).reshape(h, w, 1) * 1.5 + 0.5)
        img += gray_img
        img = np.clip(img + snow_layer + np.rot90(snow_layer, k=2), 0, 1) * 255
        img = Image.fromarray(img.astype(np.uint8))
        if isgray:
            img = ImageOps.grayscale(img)

        return img


class Rain:
    """
    Rain operation. Taken from https://github.com/roatienza/straug/blob/main/straug/weather.py
    """
    def __init__(self, rng=None):
        self.rng = np.random.default_rng() if rng is None else rng

    def __call__(self, img, mag=-1, prob=1.):
        if self.rng.uniform(0, 1) > prob:
            return img

        img = img.copy()
        w, h = img.size
        n_channels = len(img.getbands())
        isgray = n_channels == 1
        line_width = self.rng.integers(1, 2)

        c = [50, 70, 90]
        if mag < 0 or mag >= len(c):
            index = 0
        else:
            index = mag
        c = c[index]

        n_rains = self.rng.integers(c, c + 20)
        slant = self.rng.integers(-60, 60)
        fillcolor = 200 if isgray else (200, 200, 200)

        draw = ImageDraw.Draw(img)
        max_length = min(w, h, 10)
        for i in range(1, n_rains):
            length = self.rng.integers(5, max_length)
            x1 = self.rng.integers(0, w - length)
            y1 = self.rng.integers(0, h - length)
            x2 = x1 + length * math.sin(slant * math.pi / 180.)
            y2 = y1 + length * math.cos(slant * math.pi / 180.)
            x2 = int(x2)
            y2 = int(y2)
            draw.line([(x1, y1), (x2, y2)], width=line_width, fill=fillcolor)

        return img
    
class Shrink:
    """
    Shrink operation. Taken from https://github.com/roatienza/straug/blob/main/straug/geometry.py
    """
    def __init__(self, rng=None):
        self.rng = np.random.default_rng() if rng is None else rng
        self.tps = cv2.createThinPlateSplineShapeTransformer()
        self.translateXAbs = TranslateXAbs(self.rng)
        self.translateYAbs = TranslateYAbs(self.rng)

    def __call__(self, img, mag=-1, prob=1.):
        if self.rng.uniform(0, 1) > prob:
            return img

        w, h = img.size
        img = np.asarray(img)
        srcpt = []
        dstpt = []

        w_33 = 0.33 * w
        w_66 = 0.66 * w

        h_50 = 0.50 * h

        p = 0

        b = [.2, .3, .4]
        if mag < 0 or mag >= len(b):
            index = 0
        else:
            index = mag
        frac = b[index]

        # left-most
        srcpt.append([p, p])
        srcpt.append([p, h - p])
        x = self.rng.uniform(frac - .1, frac) * w_33
        y = self.rng.uniform(frac - .1, frac) * h_50
        dstpt.append([p + x, p + y])
        dstpt.append([p + x, h - p - y])

        # 2nd left-most 
        srcpt.append([p + w_33, p])
        srcpt.append([p + w_33, h - p])
        dstpt.append([p + w_33, p + y])
        dstpt.append([p + w_33, h - p - y])

        # 3rd left-most 
        srcpt.append([p + w_66, p])
        srcpt.append([p + w_66, h - p])
        dstpt.append([p + w_66, p + y])
        dstpt.append([p + w_66, h - p - y])

        # right-most 
        srcpt.append([w - p, p])
        srcpt.append([w - p, h - p])
        dstpt.append([w - p - x, p + y])
        dstpt.append([w - p - x, h - p - y])

        n = len(dstpt)
        matches = [cv2.DMatch(i, i, 0) for i in range(n)]
        dst_shape = np.asarray(dstpt).reshape((-1, n, 2))
        src_shape = np.asarray(srcpt).reshape((-1, n, 2))
        self.tps.estimateTransformation(dst_shape, src_shape, matches)
        img = self.tps.warpImage(img, borderValue=(255,255,255))
        img = Image.fromarray(img)

        if self.rng.uniform(0, 1) < 0.5:
            img = self.translateXAbs(img, val=x)
        else:
            img = self.translateYAbs(img, val=y)

        return img


class Rotate:
    """
    Rotate operation. Taken from https://github.com/roatienza/straug/blob/main/straug/geometry.py
    """
    def __init__(self, square_side=224, rng=None):
        self.side = square_side
        self.rng = np.random.default_rng() if rng is None else rng

    def __call__(self, img, iscurve=False, mag=-1, prob=1.):
        if self.rng.uniform(0, 1) > prob:
            return img

        w, h = img.size

        if h != self.side or w != self.side:
            img = img.resize((self.side, self.side), Image.BICUBIC)

        angle = self.rng.uniform(0, 15)
        if self.rng.uniform(0, 1) < 0.5:
            angle = -angle

        img = img.rotate(angle=angle, resample=Image.BICUBIC, expand=not iscurve, fillcolor=255)
        img = img.resize((w, h), Image.BICUBIC)

        return img
    
class TranslateXAbs:
    """
    TranslateXABS operation used for Shrink. Taken from https://github.com/roatienza/straug/blob/main/straug/geometry.py
    """
    def __init__(self, rng=None):
        self.rng = np.random.default_rng() if rng is None else rng

    def __call__(self, img, val=0, prob=1.):
        if self.rng.uniform(0, 1) > prob:
            return img

        v = self.rng.uniform(0, val)

        if self.rng.uniform(0, 1) > 0.5:
            v = -v
        return img.transform(img.size, Image.AFFINE, (1, 0, v, 0, 1, 0), fillcolor=255)


class TranslateYAbs:
    """
    TranslateYABS operation used for Shrink. Taken from https://github.com/roatienza/straug/blob/main/straug/geometry.py
    """
    def __init__(self, rng=None):
        self.rng = np.random.default_rng() if rng is None else rng

    def __call__(self, img, val=0, prob=1.):
        if self.rng.uniform(0, 1) > prob:
            return img

        v = self.rng.uniform(0, val)

        if self.rng.uniform(0, 1) > 0.5:
            v = -v
        return img.transform(img.size, Image.AFFINE, (1, 0, 0, 0, 1, v), fillcolor=255)
    

class RandomErasing(object):
    '''
    Class that performs Random Erasing in Random Erasing Data Augmentation. Taken from https://github.com/zhunzhong07/Random-Erasing/blob/master/utils/transforms.py 
    -------------------------------------------------------------------------------------
    probability: The probability that the operation will be performed.
    sl: min erasing area
    sh: max erasing area
    r1: min aspect ratio
    mean: erasing value
    -------------------------------------------------------------------------------------
    '''
    def __init__(self, sl = 0.02, r1 = 0.3):
        self.sl = sl
        self.r1 = r1
       
    def __call__(self, img, mag=-1, prob=1.):

        if random.uniform(0, 1) > prob:
            return img
        
        shs = [0.2, 0.3, 0.4]
        if mag < 0 or mag >= len(shs):
            index = 0
        else:
            index = mag
        sh = shs[index]

        for attempt in range(100):
            area = img.shape[0] * img.shape[1]
       
            target_area = random.uniform(self.sl, sh) * area
            aspect_ratio = random.uniform(self.r1, 1/self.r1)

            h = int(round(math.sqrt(target_area * aspect_ratio)))
            w = int(round(math.sqrt(target_area / aspect_ratio)))

            if w < img.shape[1] and h < img.shape[0]:
                x1 = random.randint(0, img.shape[0] - h)
                y1 = random.randint(0, img.shape[1] - w)
                img[x1:x1+h, y1:y1+w] = 255
                return img

        return img
    


class Erosion:
    """
    Erotion operation.
    """
    def __init__(self, kernel_size=(3, 3), rng=None):
        self.kernel_size = kernel_size
        self.rng = np.random.default_rng() if rng is None else rng

    def __call__(self, img, mag=-1, prob=1.):
        if self.rng.uniform(0, 1) > prob:
            return img
        
        kernel_sizes = [(3, 3), (5, 5), (7, 7)]
        if mag < 0 or mag >= len(kernel_sizes):
            index = 0
        else:
            index = mag
        kernel_size = kernel_sizes[index]

        kernel = np.ones(kernel_size, np.uint8)
        
        img = cv2.erode(np.array(img), kernel, iterations=1)
        return img
    


class Dilation:
    """
    Dilation operation.
    """
    def __init__(self, kernel_size=(3, 3), rng=None):
        self.kernel_size = kernel_size
        self.rng = np.random.default_rng() if rng is None else rng

    def __call__(self, img, mag=-1, prob=1.):
        if self.rng.uniform(0, 1) > prob:
            return img
        
        kernel_sizes = [(3, 3), (5, 5), (7, 7)]
        if mag < 0 or mag >= len(kernel_sizes):
            index = 0
        else:
            index = mag
        kernel_size = kernel_sizes[index]

        kernel = np.ones(kernel_size, np.uint8)
        
        img = cv2.dilate(np.array(img), kernel, iterations=1)
        return img
    

class TilingAndCorruption:
    """
    TilingAndCorruption (TACO) operation.
    """
    def __init__(self, rng=None):
        self.rng = np.random.default_rng() if rng is None else rng

    def __call__(self, img, mag=-1, prob=1.0):
        if self.rng.uniform(0, 1) > prob:
            return img
        
        corruption_probs = [0.1, 0.2, 0.3]
        if mag < 0 or mag >= len(corruption_probs):
            index = 0
        else:
            index = mag
        corruption_prob = corruption_probs[index]

        s = img.shape
        max_w = img.shape[0]//3
        max_h = max_w // (img.shape[0] // img.shape[1])
        mytaco = Taco(cp_vertical=corruption_prob,
                    cp_horizontal=corruption_prob,
                    max_tw_vertical=max_w,
                    min_tw_vertical=max_w//10,
                    max_tw_horizontal=max_h,
                    min_tw_horizontal=max_h//10
                    )
        
        r = np.random.randint(3)
        if r == 0: img = mytaco.apply_vertical_taco(img, corruption_type="white")
        elif r == 1: img = mytaco.apply_horizontal_taco(img, corruption_type="white")
        else: img = mytaco.apply_taco(img, corruption_type="white")
        img = cv2.resize(img, (s[1],s[0]))
        
        return Image.fromarray(img.astype(np.uint8))