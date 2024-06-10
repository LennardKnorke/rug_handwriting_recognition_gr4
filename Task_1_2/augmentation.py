import numpy as np
from utils import *
import cv2
import torch
import torch.nn.functional as F
from tacobox import Taco
# from molesq import ImageTransformer

from PIL import Image

from ops import SaltNoise, Rotate, Shrink, Snow, Rain, RandomErasing, Erosion, Dilation, TilingAndCorruption

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
# torch.autograd.set_detect_anomaly(True)

##############################################
# CLASSES AND FUNCTIONS FOR THE AUGMENTATION SECTION
##############################################
class WarpMLS:
    """
    Moving Least Squares transformation on an image, given some number of source points and distance points.
    Taken from https://github.com/RubanSeven/Text-Image-Augmentation-python/blob/master/warp_mls.py
    """
    def __init__(self, src, src_pts, dst_pts, dst_w, dst_h, trans_ratio=1.):
        self.src = src
        self.src_pts = src_pts
        self.dst_pts = dst_pts
        self.pt_count = len(self.dst_pts)
        self.dst_w = dst_w
        self.dst_h = dst_h
        self.trans_ratio = trans_ratio
        self.grid_size = 100
        self.rdx = np.zeros((self.dst_h, self.dst_w))
        self.rdy = np.zeros((self.dst_h, self.dst_w))

    @staticmethod
    def __bilinear_interp(x, y, v11, v12, v21, v22):
        return (v11 * (1 - y) + v12 * y) * (1 - x) + (v21 * (1 - y) + v22 * y) * x

    def generate(self):
        self.calc_delta()
        return self.gen_img()

    def calc_delta(self):
        w = np.zeros(self.pt_count, dtype=np.float32)

        if self.pt_count < 2:
            return

        i = 0
        while 1:
            if self.dst_w <= i < self.dst_w + self.grid_size - 1:
                i = self.dst_w - 1
            elif i >= self.dst_w:
                break

            j = 0
            while 1:
                if self.dst_h <= j < self.dst_h + self.grid_size - 1:
                    j = self.dst_h - 1
                elif j >= self.dst_h:
                    break

                sw = 0
                swp = np.zeros(2, dtype=np.float32)
                swq = np.zeros(2, dtype=np.float32)
                new_pt = np.zeros(2, dtype=np.float32)
                cur_pt = np.array([i, j], dtype=np.float32)

                k = 0
                for k in range(self.pt_count):
                    if i == self.dst_pts[k][0] and j == self.dst_pts[k][1]:
                        break

                    w[k] = 1. / ((i - self.dst_pts[k][0]) * (i - self.dst_pts[k][0]) +
                                 (j - self.dst_pts[k][1]) * (j - self.dst_pts[k][1]))

                    sw += w[k]
                    swp = swp + w[k] * np.array(self.dst_pts[k])
                    swq = swq + w[k] * np.array(self.src_pts[k])

                if k == self.pt_count - 1:
                    pstar = 1 / sw * swp
                    qstar = 1 / sw * swq

                    miu_s = 0
                    for k in range(self.pt_count):
                        if i == self.dst_pts[k][0] and j == self.dst_pts[k][1]:
                            continue
                        pt_i = self.dst_pts[k] - pstar
                        miu_s += w[k] * np.sum(pt_i * pt_i)

                    cur_pt -= pstar
                    cur_pt_j = np.array([-cur_pt[1], cur_pt[0]])

                    for k in range(self.pt_count):
                        if i == self.dst_pts[k][0] and j == self.dst_pts[k][1]:
                            continue

                        pt_i = self.dst_pts[k] - pstar
                        pt_j = np.array([-pt_i[1], pt_i[0]])

                        tmp_pt = np.zeros(2, dtype=np.float32)
                        tmp_pt[0] = np.sum(pt_i * cur_pt) * self.src_pts[k][0] - \
                                    np.sum(pt_j * cur_pt) * self.src_pts[k][1]
                        tmp_pt[1] = -np.sum(pt_i * cur_pt_j) * self.src_pts[k][0] + \
                                    np.sum(pt_j * cur_pt_j) * self.src_pts[k][1]
                        tmp_pt *= (w[k] / miu_s)
                        new_pt += tmp_pt

                    new_pt += qstar
                else:
                    new_pt = self.src_pts[k]

                self.rdx[j, i] = new_pt[0] - i
                self.rdy[j, i] = new_pt[1] - j

                j += self.grid_size
            i += self.grid_size

    def gen_img(self):
        src_h, src_w = self.src.shape[:2]
        dst = np.zeros_like(self.src, dtype=np.float32)

        for i in np.arange(0, self.dst_h, self.grid_size):
            for j in np.arange(0, self.dst_w, self.grid_size):
                ni = i + self.grid_size
                nj = j + self.grid_size
                w = h = self.grid_size
                if ni >= self.dst_h:
                    ni = self.dst_h - 1
                    h = ni - i + 1
                if nj >= self.dst_w:
                    nj = self.dst_w - 1
                    w = nj - j + 1

                di = np.reshape(np.arange(h), (-1, 1))
                dj = np.reshape(np.arange(w), (1, -1))
                delta_x = self.__bilinear_interp(di / h, dj / w,
                                                 self.rdx[i, j], self.rdx[i, nj],
                                                 self.rdx[ni, j], self.rdx[ni, nj])
                delta_y = self.__bilinear_interp(di / h, dj / w,
                                                 self.rdy[i, j], self.rdy[i, nj],
                                                 self.rdy[ni, j], self.rdy[ni, nj])
                nx = j + dj + delta_x * self.trans_ratio
                ny = i + di + delta_y * self.trans_ratio
                nx = np.clip(nx, 0, src_w - 1)
                ny = np.clip(ny, 0, src_h - 1)
                nxi = np.array(np.floor(nx), dtype=np.int32)
                nyi = np.array(np.floor(ny), dtype=np.int32)
                nxi1 = np.array(np.ceil(nx), dtype=np.int32)
                nyi1 = np.array(np.ceil(ny), dtype=np.int32)

                if len(self.src.shape) == 3:
                    x = np.tile(np.expand_dims(ny - nyi, axis=-1), (1, 1, 3))
                    y = np.tile(np.expand_dims(nx - nxi, axis=-1), (1, 1, 3))
                else:
                    x = ny - nyi
                    y = nx - nxi
                dst[i:i + h, j:j + w] = self.__bilinear_interp(x,
                                                               y,
                                                               self.src[nyi, nxi],
                                                               self.src[nyi, nxi1],
                                                               self.src[nyi1, nxi],
                                                               self.src[nyi1, nxi1]
                                                               )

        dst = np.clip(dst, 0, 255)
        dst = np.array(dst, dtype=np.uint8)

        return dst


def sample_radius(radius, max_radius=False):
    """
    Sample a random radius within a limit (radius), or return max_radius when given.
    """
    if max_radius: return radius
    else: return np.random.randint(radius)
    
def distort(src, n_patches, radius, movement, vertical=True, return_points=False, max_radius=False, radii=None):
    """
    Initialize source points on an image according to the number of patches, move these points to distance points
    according to the maximum radius and moving state, and augment image with MLS. If max radius is given, the
    movements will always use the radius threshold. If a list of radii is given, these will be used. If return_points
    is true, the source and destination points will be returned besides the distored image, and otherwise the sampled
    radii. Adapted from https://github.com/RubanSeven/Text-Image-Augmentation-python/blob/master/warp_mls.py
    """
    movement[movement==0] = -1
    if radii is None:
        radii = []
        for _ in range(2*(n_patches+1)*2):
            radii.append(sample_radius(radius, max_radius))
    radii_copy = radii.copy()
    # if movement[0][0] == -1 and movement[0][1] == -1: radii[:] = [100 for _ in radii]
    img_h, img_w = src.shape[:2]

    if vertical: cut = img_h // n_patches
    else: cut = img_w // n_patches

    src_pts = list()
    dst_pts = list()

    src_pts.append([0, 0])
    src_pts.append([img_w, 0])
    src_pts.append([img_w, img_h])
    src_pts.append([0, img_h])
    if n_patches == 4:
        src_pts.append([img_w//2, img_h//2])

    dst_pts.append([radii.pop()*movement[0][0], radii.pop()*movement[0][1]])
    dst_pts.append([img_w + radii.pop()*movement[1][0], radii.pop()*movement[1][1]])
    dst_pts.append([img_w + radii.pop()*movement[2][0], img_h + radii.pop()*movement[2][1]])
    dst_pts.append([radii.pop()*movement[3][0], img_h + radii.pop()*movement[3][1]])
    if n_patches == 4:
        dst_pts.append([img_w//2 + radii.pop()*movement[4][0], img_h//2 + radii.pop()*movement[3][1]])
    else:
        # half_radius = radius * 0.5
        half_radius = 0
        p_idx = 3
        for cut_idx in np.arange(1, n_patches, 1):
            if vertical:
                src_pts.append([0, cut * cut_idx])
                src_pts.append([img_w, cut * cut_idx])
            else:
                src_pts.append([cut * cut_idx, 0])
                src_pts.append([cut * cut_idx, img_h])

            p_idx += 1
            if vertical: dst_pts.append([(radii.pop() - half_radius)*movement[p_idx][0],
                            cut * cut_idx + (radii.pop() - half_radius)*movement[p_idx][1]])
            else: dst_pts.append([cut * cut_idx + (radii.pop() - half_radius)*movement[p_idx][0],
                            (radii.pop() - half_radius)*movement[p_idx][1]])
            p_idx += 1
            if vertical: dst_pts.append([img_w + (radii.pop() - half_radius)*movement[p_idx][0],
                            cut * cut_idx + (radii.pop() - half_radius)*movement[p_idx][1]])
            else: dst_pts.append([cut * cut_idx + (radii.pop() - half_radius)*movement[p_idx][0],
                            img_h + (radii.pop() - half_radius)*movement[p_idx][1]])

    trans = WarpMLS(src, src_pts, dst_pts, img_w, img_h)
    dst = trans.generate()
    # trans = ImageTransformer(src, np.array(src_pts), np.array(dst_pts), color_dim=2, interp_order=2)
    # dst = trans.deform_viewport()
    if return_points: return dst, src_pts, dst_pts
    return dst, radii_copy

def distort_batch(images, n_patches, radius, S, radii_list=None):
    """
    Disort a batch of images with elastic morphing based on the moving state array S.
    """
    images = torch.squeeze(images, 1)
    aug_images = torch.empty_like(images).detach().cpu()
    images = images.detach().cpu().numpy()
    S_cpu = S.detach().cpu().numpy()
    new_radii_list = []
    for i in range(len(images)):
        if radii_list is None:
            distort_img, new_radii = distort(images[i], n_patches, radius, S_cpu[i])
            new_radii_list.append(new_radii)
        else:
            distort_img, _ = distort(images[i], n_patches, radius, S_cpu[i], radii=radii_list[i])
        aug_images[i] = torch.from_numpy(distort_img)

    aug_images = aug_images.reshape((len(images), 1, CHARACTER_HEIGHT, CHARACTER_WIDTH))
    if new_radii_list: return aug_images, new_radii_list
    return aug_images

def augment_data(images, n_patches, radius, agent=None):
    """
    Augment images with elastic morphing.
    @param images: Images to augment
    @param n_patches: Number of patches
    @param radius: Radius threshold
    @param agent: Augmentation agent.
    @return Augmented/distorted images, randomly augmented images (used for learning the augmentation agent),
            outputs of the agent, moving state of the distortion, moving state of the random distortion.
            If agent is not provided, returns only randomly augmented images
    """
    images = images.to(device)

    if agent is not None:
        agent_outputs = agent(images)
        S = torch.max(agent_outputs, 3).indices

        # sample instead of taking max
        # S_flat = agent_outputs.view(-1, 2)
        # indices = torch.multinomial(S_flat, 1)
        # S = indices.view(agent_outputs.size(0), agent_outputs.size(1), agent_outputs.size(3))

        S2 = S.detach().clone().cpu()
        rev_points = np.random.randint(S.shape[1], size=S.shape[0])
        rev_dirs = np.random.randint(2, size=S.shape[0])
        mask = torch.zeros_like(S2, dtype=torch.bool)
        mask[torch.arange(rev_points.shape[0]), rev_points, rev_dirs] = True
        S2 = torch.where(mask, 1 - S2, S2)
        aug_S, radii = distort_batch(images, n_patches, radius, S.clone())
        aug_S2 = distort_batch(images, n_patches, radius, S2.clone(), radii_list=radii)
        return aug_S, aug_S2, agent_outputs, S, S2
    
    else:
        S = np.random.randint(2,size=images.shape[0]*2*(n_patches+1)*2).reshape((images.shape[0], 2*(n_patches+1), 2))
        S = torch.from_numpy(S)
        aug_S, _ = distort_batch(images, n_patches, radius, S)
        return aug_S

def learning_agent_loss(outputs, outputs_S2, labels, agent_outputs, S, S2):
    S = S.to(device)
    S2 = S2.to(device)

    # use "edit distance" 1 or 0
    # recognizer_loss_S = torch.max(outputs, 1).indices != torch.max(labels, 1).indices
    # recognizer_loss_S = recognizer_loss_S.int()
    # recognizer_loss_S2 = torch.max(outputs_S2, 1).indices != torch.max(labels, 1).indices
    # recognizer_loss_S2 = recognizer_loss_S2.int()

    # use recognizer loss
    recognizer_loss_S = F.cross_entropy(outputs, labels, reduction='none')
    recognizer_loss_S2 = F.cross_entropy(outputs_S2, labels, reduction='none')

    # true = S2.unsqueeze(-1).expand(-1,-1,-1,2)
    # mask2 = torch.arange(2).expand_as(true)
    # true = torch.where(true==mask2, 1, 0)
    # true = true.view(agent_outputs.size(0),-1)
 
    # mask = recognizer_loss_S <= recognizer_loss_S2
    # mask = mask.unsqueeze(-1).unsqueeze(-1).expand_as(S2)
    # mask = mask & torch.max(agent_outputs,3).indices != S2
    # S2 = torch.where(mask, 1 - S2, S2)
    # S2 = S2.view(-1)
    # agent_outputs = agent_outputs.view(-1, 2)
    # loss = F.nll_loss(torch.log(agent_outputs + 1e-20), S2)
 
    # print(agent_outputs[0][0].tolist())
    # if (S[0][0][0] != 0 or S[0][0][1] != 0) and S2[0][0][0] == 0 and S2[0][0][1] == 0:
    #     print(S[0][0].tolist(), S2[0][0].tolist())
    #     print("S", round(float(recognizer_loss_S.float().mean()), 3))
    #     print("S2", round(float(recognizer_loss_S2.float().mean()), 3))
    rev_mask = S != S2
    rev = rev_mask.any(-1).max(-1).indices
    S_rev = S[torch.arange(rev.size(0)), rev]
    S2_rev = S2[torch.arange(rev.size(0)), rev]
    agent_outputs = agent_outputs[torch.arange(rev.size(0)), rev]
    mask = recognizer_loss_S <= recognizer_loss_S2
    mask = mask.unsqueeze(-1).expand_as(S2_rev)
    true = torch.where(mask, S2_rev, S_rev)
    true = true.view(-1)
    agent_outputs = agent_outputs.view(-1, 2)
    loss = F.nll_loss(torch.log(agent_outputs + 1e-20), true)
    return loss

def train(outputs, outputs_S2, labels, agent_opt, agent_outputs, S, S2):
    """
    Train the learning augmentation agent.
    @param outputs: Outputs of the recognizer on the augmented images
    @param outputs_S2: Outputs of the recognizer on the randomly augmented images
    @param labels: True image classes
    @param agent_opt: Agent optimizer
    @param agent_outputs: Agent outputs from this epoch
    @param S: Distortion moving state
    @param S2: Random distortion moving state
    @return Learning agent loss.
    """
    criterion = learning_agent_loss
    loss = criterion(outputs, outputs_S2, labels, agent_outputs, S, S2)
    if not torch.isnan(loss):
        agent_opt.zero_grad()
        loss.backward()
        agent_opt.step()
        return loss.item()
    return 0.0

def draw_augment_arrows(img, src_pts, dst_pts, radius):
    """
    Preview distortion with arrows from source points to destination points, displayed using the maximum radius.
    """
    img = cv2.cvtColor(img,cv2.COLOR_GRAY2RGB)
    img = cv2.copyMakeBorder(img, radius, radius, radius, radius, cv2.BORDER_CONSTANT)
    for j in range(len(src_pts)):
        img = cv2.arrowedLine(img, (src_pts[j][0]+radius,src_pts[j][1]+radius), (dst_pts[j][0]+radius,dst_pts[j][1]+radius), color=(0, 0, 255), thickness=4)
    return img
    
def elastic_demo(img_file, n_patches, radius):
    im = file_to_img(img_file)
    distort_img_list = list()
    for _ in range(100):
        if n_patches == 4: S = np.random.randint(2,size=5*2).reshape((5, 2))
        else: S = np.random.randint(2,size=2*(n_patches+1)*2).reshape((2*(n_patches+1), 2))
        distort_img, src_pts, dst_pts = distort(im, n_patches, radius, S, return_points=True)
        distort_img = draw_augment_arrows(distort_img, src_pts, dst_pts, radius)
        distort_img_list.append(distort_img)
    create_gif(distort_img_list, r'img/distort.gif')

class RandomAug(object):
    def __init__(self, N, M):
        self.N = N
        self.M = M
        
        self.aug_list = []
        self.aug_list.extend([Rotate(), Shrink()])
        self.aug_list.extend([SaltNoise()])
        self.aug_list.extend([Snow(), Rain()])
        self.aug_list.extend([RandomErasing()])
        self.aug_list.extend([Erosion(), Dilation()])
        self.aug_list.extend([TilingAndCorruption()]) 

        self.mag_range = np.random.randint(0, 3)

    def __call__(self, img):
        for _ in range(self.N):
            op = np.random.randint(0, len(self.aug_list))
            if op <= 4 and not isinstance(img, Image.Image): 
                img = Image.fromarray(img)
            elif op > 4 and isinstance(img, Image.Image):
                img = np.array(img)
            img = self.aug_list[op](img, mag = self.M, prob = 1)
            img = np.array(img)

        return img
    
def randaug_data(images, randaug_sampler):
    """
    Augment images with RandAugment.
    @param images: Images to augment
    @param agent: Augmentation agent.
    @return Augmented images
    """
    images = torch.squeeze(images, 1)
    aug_images = torch.empty_like(images).detach().cpu()
    images = images.detach().cpu().numpy()
    for i in range(len(images)):
        distort_img = Image.fromarray(images[i])
        distort_img = randaug_sampler(distort_img)
        distort_img = np.array(distort_img)
        distort_img = cv2.cvtColor(distort_img, cv2.COLOR_RGB2BGR)
        distort_img = cv2.cvtColor(distort_img, cv2.COLOR_BGR2GRAY)
        distort_img = cv2.threshold(distort_img, 120, 255, cv2.THRESH_BINARY)[1]
        aug_images[i] = torch.from_numpy(distort_img)

    aug_images = aug_images.reshape((len(images), 1, CHARACTER_HEIGHT, CHARACTER_WIDTH))
    return aug_images
    
def randaug_demo(N, M):
    randaug_1 = RandomAug(N, M)
    
    pth = "img//randaug"
    img_file = "img//segmented//Alef//navis-QIrug-Qumran_extr09_0001-line-008-y1=400-y2=515-zone-HUMAN-x=1650-y=0049-w=0035-h=0042-ybas=0027-nink=631-segm=COCOS5cocos.pgm"
    os.makedirs(pth, exist_ok=True)

    img = file_to_img(img_file)
    
    augmented_img_1 = randaug_1(img)

    # Save images to compare before and after augmentation.
    result = cv2.cvtColor(np.hstack((np.array(img), np.array(augmented_img_1))), cv2.COLOR_RGB2BGR)
    result = cv2.cvtColor(result, cv2.COLOR_BGR2GRAY)
    result = cv2.threshold(result, 120, 255, cv2.THRESH_BINARY)[1]
    cv2.imwrite(os.path.join(pth, 'randaug.png'), result)

def taco_demo(img, cp):
    s = img.shape
    max_w = img.shape[0]//3
    max_h = max_w // (img.shape[0] // img.shape[1])
    print(max_w,max_h)
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
        augmented_img = cv2.resize(augmented_img, (s[1],s[0]))  # Resize the image
        distort_img_list.append(augmented_img)
    create_gif(distort_img_list, r'img/distort.gif')

    # mytaco.visualize(augmented_img)

##############################################
#
##############################################
if __name__ == '__main__':
    print("Running the augmentation script only")
    # elastic_demo("img//segmented//Alef//navis-QIrug-Qumran_extr09_0001-line-008-y1=400-y2=515-zone-HUMAN-x=1650-y=0049-w=0035-h=0042-ybas=0027-nink=631-segm=COCOS5cocos.pgm",
        #  2, 10)

    randaug_demo(3, 1)

    # img = file_to_img("img//segmented//Alef//navis-QIrug-Qumran_extr09_0001-line-008-y1=400-y2=515-zone-HUMAN-x=1650-y=0049-w=0035-h=0042-ybas=0027-nink=631-segm=COCOS5cocos.pgm")
    # taco_demo(img, 0.25)