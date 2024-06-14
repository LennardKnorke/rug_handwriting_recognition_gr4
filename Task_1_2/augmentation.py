import numpy as np
from utils import *
from learn_to_augment import *
import cv2
import torch
from tacobox import Taco
import argparse
from PIL import Image

from ops import SaltNoise, Rotate, Shrink, Snow, Rain, RandomErasing, Erosion, Dilation, TilingAndCorruption

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")


def parse_args() -> argparse.Namespace:
    """Parse the command-line arguments"""

    parser = argparse.ArgumentParser(description="")

    parser.add_argument(
        'input_path',
        type=str,
        help='Path to the image to augment.'
    )

    parser.add_argument(
        '-a',
        '--augment',
        type=str,
        choices=['elastic', 'randaug', 'taco'],
        nargs='*',
        default=['randaug'],
        help='Augmentation method to demonstrate: Elastic morphing, RandomAugment, or TilingAndCorruption.'
    )

    parser.add_argument(
        '-p',
        '--patches',
        type=int,
        default=2,
        help='Number of patches to use for elastic morphing.'
    )

    parser.add_argument(
        '-r',
        '--radius',
        type=int,
        default=10,
        help='Radius to use for elastic morphing.'
    )

    parser.add_argument(
        '-N',
        '--n_randaug',
        type=int,
        default=3,
        help='Number of augmentations to apply sequentially for RandomAugment.'
    )

    parser.add_argument(
        '-M',
        '--m_randaug',
        type=int,
        default=1,
        choices=range(3),
        help='Magnitude (0-2) of augmentations for RandomAugment.'
    )

    parser.add_argument(
        '-cp',
        '--corruption_probability',
        type=int,
        default=0.25,
        choices=range(3),
        help='Corruption probability for TACO.'
    )

    args = parser.parse_args()

    if not os.path.exists(args.input_path):
        parser.error('The input image does not exist.')

    return args

class RandomAug(object):
    """
    Object to sample augmentations from when using RandomAugment.
    """
    def __init__(self, N, M):
        """
        Initialize the RandomAug object.
        @param N: Number of augmentations to apply sequentially.
        @param M: Magnitude (0-2) of augmentations.
        """
        self.N = N
        self.M = M
        
        self.aug_list = [
            Rotate(),
            Shrink(),
            SaltNoise(),
            Snow(),
            Rain(),
            RandomErasing(),
            Erosion(),
            Dilation(),
            TilingAndCorruption()
        ]

    def __call__(self, img):
        """
        Augment an image using RandomAugment
        @param img: Image to augment
        @return Augmented image
        """
        for i in range(self.N):
            if i == 0: op = 0
            if i == 1: op = 8
            else: op = np.random.randint(1, len(self.aug_list)-1)

            if op <= 4 and not isinstance(img, Image.Image): 
                img = Image.fromarray(img)
            elif op > 4 and isinstance(img, Image.Image):
                img = np.array(img)

            img = self.aug_list[op](img, mag =self.M, prob=1)
            img = np.array(img)

        return img
    
def randaug_data(images, randaug_sampler):
    """
    Augment images with RandAugment.
    @param images: Images to augment
    @param randaug_sampler: RandAug object to sample augmentations from.
    @return Augmented images
    """
    images = torch.squeeze(images, 1)
    aug_images = torch.empty_like(images).detach().cpu()
    images = images.detach().cpu().numpy()

    for i in range(len(images)):
        distort_img = Image.fromarray(images[i])
        distort_img = randaug_sampler(distort_img)
        distort_img = cv2.cvtColor(distort_img, cv2.COLOR_RGB2BGR)
        distort_img = cv2.cvtColor(distort_img, cv2.COLOR_BGR2GRAY)
        distort_img = cv2.threshold(distort_img, 120, 255, cv2.THRESH_BINARY)[1]
        aug_images[i] = torch.from_numpy(distort_img)

    aug_images = aug_images.reshape((len(images), 1, CHARACTER_HEIGHT, CHARACTER_WIDTH))
    return aug_images

def draw_augment_arrows(img, src_pts, dst_pts, radius):
    """
    Preview distortion with arrows from source points to destination points, displayed using the maximum radius.
    @param img: Image to draw on
    @param src_pts: Coordinates of source fiducial points
    @param dst_pts: Coordinates of destination fiducial points
    @param radius: Maximum radius for moving fiducial points.
    @return Image with arrows
    """
    img = cv2.cvtColor(img,cv2.COLOR_GRAY2RGB)
    img = cv2.copyMakeBorder(img, radius, radius, radius, radius, cv2.BORDER_CONSTANT)
    for j in range(len(src_pts)):
        img = cv2.arrowedLine(img,
                              (src_pts[j][0]+radius,src_pts[j][1]+radius),
                              (dst_pts[j][0]+radius,dst_pts[j][1]+radius),
                              color=(0, 0, 255),
                              thickness=4
                            )
    return img
    
def elastic_demo(img, n_patches, radius):
    """
    Demonstrate Elastic morphing by creating a gif of 100 different augmentations.
    @param img: Image to augment
    @param n_patches: Number of patches.
    @param radius: Maximum radius for moving fiducial points.
    """
    distort_img_list = list()
    for _ in range(100):
        S = np.random.randint(2,size=2*(n_patches+1)*2).reshape((2*(n_patches+1), 2))
        distort_img, src_pts, dst_pts = distort(img, n_patches, radius, S, return_points=True)
        distort_img = draw_augment_arrows(distort_img, src_pts, dst_pts, radius)
        distort_img_list.append(distort_img)
    create_gif(distort_img_list, r'elastic_demo.gif')
    
def randaug_demo(img, N, M):
    """
    Demonstrate RandomAugment by creating a gif of 100 different augmentations.
    @param img: Image to augment
    @param N: Number of augmentations to apply sequentially.
    @param M: Magnitude (0-2) of augmentations.
    """
    randaug_1 = RandomAug(N, M)

    distort_img_list = list()
    for _ in range(100):
        augmented_img = randaug_1(img)
        distort_img_list.append(augmented_img)
    create_gif(distort_img_list, r'randaug_demo.gif')

    # Save images to compare before and after augmentation.
    # result = cv2.cvtColor(np.hstack((np.array(img), np.array(augmented_img_1))), cv2.COLOR_RGB2BGR)
    # result = cv2.cvtColor(result, cv2.COLOR_BGR2GRAY)
    # result = cv2.threshold(result, 120, 255, cv2.THRESH_BINARY)[1]
    # cv2.imwrite(os.path.join(pth, 'randaug.png'), result)


def taco_demo(img, cp):
    """
    Demonstrate TilingAndCorrpution by creating a gif of 100 different augmentations.
    @param img: Image to augment
    @param cp: Corruption probability
    """
    s = img.shape
    max_w = img.shape[0]//3
    max_h = max_w // (img.shape[0] // img.shape[1])

    mytaco = Taco(cp_vertical=cp,
                cp_horizontal=cp,
                max_tw_vertical=max_w,
                min_tw_vertical=max_w//10,
                max_tw_horizontal=max_h,
                min_tw_horizontal=max_h//10
                )
    
    distort_img_list = list()
    for _ in range(100):
        r = np.random.randint(3)
        if r == 0: augmented_img = mytaco.apply_vertical_taco(img, corruption_type="white")
        elif r == 1: augmented_img = mytaco.apply_horizontal_taco(img, corruption_type="white")
        else: augmented_img = mytaco.apply_taco(img, corruption_type="white")

        augmented_img = cv2.resize(augmented_img, (s[1],s[0])) # The ouput size sometimes differs slightly
        # mytaco.visualize(augmented_img)
        distort_img_list.append(augmented_img)
    create_gif(distort_img_list, r'taco_demo.gif')


def main():
    """
    When running augmentation.py directly, perform a demonstration of one of the augmentation techniques.
    """
    args = parse_args()
    img = file_to_img(args.input_path)

    if args.aug == "elastic":
        elastic_demo(img, args.p, args.r)

    elif args.aug == "randaug":
        randaug_demo(img, args.N, args.M)

    elif args.aug == "taco":
        taco_demo(img, args.cp)

if __name__ == '__main__':
    main()