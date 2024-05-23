import numpy as np
from utils import *
import cv2
import torch
import torch.nn.functional as F
# from molesq import ImageTransformer

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

##############################################
# CLASSES AND FUNCTIONS FOR THE AUGMENTATION SECTION
##############################################
class WarpMLS:
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

def distort(src, n_patches, radius, movement):
    img_h, img_w = src.shape[:2]

    cut = img_w // n_patches

    src_pts = list()
    dst_pts = list()

    src_pts.append([0, 0])
    src_pts.append([img_w, 0])
    src_pts.append([img_w, img_h])
    src_pts.append([0, img_h])

    dst_pts.append([np.random.randint(radius)*movement[0][0], np.random.randint(radius)*movement[0][1]])
    dst_pts.append([img_w - np.random.randint(radius)*movement[1][0], np.random.randint(radius)*movement[1][1]])
    dst_pts.append([img_w - np.random.randint(radius)*movement[2][0], img_h - np.random.randint(radius)*movement[2][1]])
    dst_pts.append([np.random.randint(radius)*movement[3][0], img_h - np.random.randint(radius)*movement[3][1]])

    # half_radius = radius * 0.5
    half_radius = radius
    p_idx = 3
    for cut_idx in np.arange(1, n_patches, 1):
        src_pts.append([cut * cut_idx, 0])
        src_pts.append([cut * cut_idx, img_h])
        p_idx += 1
        dst_pts.append([cut * cut_idx + (np.random.randint(radius) - half_radius)*movement[p_idx][0],
                        (np.random.randint(radius) - half_radius)*movement[p_idx][1]])
        p_idx += 1
        dst_pts.append([cut * cut_idx + (np.random.randint(radius) - half_radius)*movement[p_idx][0],
                        img_h + (np.random.randint(radius) - half_radius)*movement[p_idx][1]])

    trans = WarpMLS(src, src_pts, dst_pts, img_w, img_h)
    dst = trans.generate()
    # trans = ImageTransformer(src, np.array(src_pts), np.array(dst_pts), color_dim=2, interp_order=2)
    # dst = trans.deform_viewport()
    return dst
    
def demo(img_file, n_patches, radius):
    im = file_to_img(img_file)
    # cv2.imshow("im_CV", im)
    distort_img_list = list()
    for i in range(100):
        S = np.random.randint(2,size=2*(n_patches+1)*2).reshape((2*(n_patches+1), 2))
        distort_img = distort(im, n_patches, radius, S)
        distort_img_list.append(distort_img)
        # cv2.imshow("distort_img", distort_img)
        # cv2.waitKey(100)
    create_gif(distort_img_list, r'img/distort.gif')

def augment_batch(images, n_patches, radius, S):
    images = torch.squeeze(images)
    aug_images = torch.empty_like(images).detach().cpu()
    images = images.detach().cpu().numpy()
    S = S.detach().cpu().numpy()
    for i in range(len(images)):
        aug_images[i] = torch.from_numpy(distort(images[i], n_patches, radius, S[i]))
    return aug_images

def augment_data(images, agent, n_patches, radius):
    images = images.to(device)
    agent_outputs = agent(images)

    S = torch.max(agent_outputs, 3).indices
    S[S==0] = -1
    S2 = S.detach().clone()
    S2_probs = torch.max(F.softmax(agent_outputs, 3), 3).values
    rev_points = np.random.randint(2, size=len(S))
    S2[:][rev_points] = -1 * S2[:][rev_points]
    S2_probs[:][rev_points] = 1 - S2_probs[:][rev_points]
    
    aug_S = augment_batch(images, n_patches, radius, S).reshape((len(images), 1, CHARACTER_HEIGHT, CHARACTER_WIDTH))
    aug_S2 = augment_batch(images, n_patches, radius, S2).reshape((len(images), 1, CHARACTER_HEIGHT, CHARACTER_WIDTH))

    return aug_S, aug_S2, S2_probs.detach()

# def augment_loss(output, target):
#     loss = torch.mean((output - target)**2)
#     return loss

def train(outputs, outputs_S2, labels, agent_opt, S2_probs):
    loss = 0.0
    for i in range(len(outputs)):
        recognizer_loss_S = RECOGNIZER_LOSS_FUNCTION(outputs[i], labels[i])
        recognizer_loss_S2 = RECOGNIZER_LOSS_FUNCTION(outputs_S2[i], labels[i])

        if recognizer_loss_S <= recognizer_loss_S2:
            S2_probs_i = torch.prod(S2_probs[i], 1)
        else:
            S2_probs_i = torch.prod(1 - S2_probs[i], 1)

        loss += -1 * torch.sum(torch.log(S2_probs_i))

    loss.requires_grad=True
    agent_opt.zero_grad()
    loss.backward()
    # torch.nn.utils.clip_grad_value_(agent.parameters(), 100)
    agent_opt.step()

##############################################
#
##############################################
if __name__ == '__main__':
    print("Running the augmentation script only")
    demo("img//segmented//Alef//navis-QIrug-Qumran_extr09_0001-line-008-y1=400-y2=515-zone-HUMAN-x=1650-y=0049-w=0035-h=0042-ybas=0027-nink=631-segm=COCOS5cocos.pgm",
         3, 20)
    pass