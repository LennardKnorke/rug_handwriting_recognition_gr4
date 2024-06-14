import numpy as np
import cv2
import matplotlib.pyplot as plt
import copy

import torch
import torch.nn.functional as F

from utils import *

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

    @param src: Image to distort
    @param n_patches: Number of patches
    @param radius: Maximum radius for moving fiducial points.
    @param movement: Moving state for the directions of movement for each fiducial point.
    @param vertical: Whether the patches should be arranged vertically
    @param return_points: Return coordinates of source and destination points
    @param max_radius: If True, always move fiducial points with the maximum radius
    @param radii: Custom array of radii to use for distortion

    @return Distorted image
    @return If return_points is False, also return an array of radii that were used for distortion
    @return If return_points is True, also return coordinates of source fiducial points
    @return If return_points is True, also return coordinates of destination fiducial points
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

    dst_pts.append([radii.pop()*movement[0][0], radii.pop()*movement[0][1]])
    dst_pts.append([img_w + radii.pop()*movement[1][0], radii.pop()*movement[1][1]])
    dst_pts.append([img_w + radii.pop()*movement[2][0], img_h + radii.pop()*movement[2][1]])
    dst_pts.append([radii.pop()*movement[3][0], img_h + radii.pop()*movement[3][1]])
    
    p_idx = 3
    for cut_idx in np.arange(1, n_patches, 1):
        if vertical:
            src_pts.append([0, cut * cut_idx])
            src_pts.append([img_w, cut * cut_idx])
        else:
            src_pts.append([cut * cut_idx, 0])
            src_pts.append([cut * cut_idx, img_h])

        p_idx += 1
        if vertical: dst_pts.append([(radii.pop())*movement[p_idx][0],
                        cut * cut_idx + (radii.pop())*movement[p_idx][1]])
        else: dst_pts.append([cut * cut_idx + (radii.pop())*movement[p_idx][0],
                        (radii.pop())*movement[p_idx][1]])
        p_idx += 1
        if vertical: dst_pts.append([img_w + (radii.pop())*movement[p_idx][0],
                        cut * cut_idx + (radii.pop())*movement[p_idx][1]])
        else: dst_pts.append([cut * cut_idx + (radii.pop())*movement[p_idx][0],
                        img_h + (radii.pop())*movement[p_idx][1]])

    trans = WarpMLS(src, src_pts, dst_pts, img_w, img_h)
    dst = trans.generate()

    if return_points: return dst, src_pts, dst_pts
    return dst, radii_copy

def distort_batch(images, words, bboxes, n_patches, radius, S, radii_list=None):
    """
    Disort a batch of images with elastic morphing based on the moving state array S. Only augment one word in each line.
    @param images: Images to distort
    @param words: Images segmented into individual words
    @param bboxes: Original bounding box coordinates of words
    @param n_patches: Number of patches
    @param radius: Radius threshold
    @param S: Moving state for the directions of movement for each fiducial point.
    @param radii_list: Custom array of radii to use for distortion

    @return Distorted images
    @return If radii_list is not given, also return an array of radii that were used for distortion
    """
    words = torch.squeeze(words, 1)
    aug_images = copy.deepcopy(images)
    words = words.detach().cpu().numpy()
    S_cpu = S.detach().cpu().numpy()

    new_radii_list = []
    for i in range(len(words)):
        if radii_list is None:
            word, new_radii = distort(words[i], n_patches, radius, S_cpu[i])
            new_radii_list.append(new_radii)
        else:
            word, _ = distort(words[i], n_patches, radius, S_cpu[i], radii=radii_list[i])

        # Place augmented word back into the line
        x, y, w, h = bboxes[i]
        word = cv2.resize(word, (w, h))
        distort_line = images[i].copy()
        distort_line[y:y+h, x:x+w] = word
        aug_images[i] = distort_line

    if new_radii_list: return aug_images, new_radii_list
    return aug_images



def augment_data(images, n_patches, radius, agent=None):
    """
    Augment images with elastic morphing. Only augment one word in each line.
    @param images: Images to augment
    @param n_patches: Number of patches
    @param radius: Radius threshold
    @param agent: Augmentation agent.
    @return Augmented/distorted images, randomly augmented images (used for learning the augmentation agent),
            outputs of the agent, moving state of the distortion, moving state of the random distortion.
            If agent is not provided, returns only randomly augmented images
    """

    # Segment lines into words
    segmented = [line_to_words(image) for image in images]
    word_ids = [np.random.randint(len(x[0])) for x in segmented]
    words, bboxes = [x[0][word_ids[i]] for i,x in enumerate(segmented)], [x[1][word_ids[i]] for i,x in enumerate(segmented)]
    words = np.array([cv2.resize(word, (100, 32)) for word in words])
    words = torch.FloatTensor(words).reshape((len(images), 1, 32, 100))
    words = words.to(DEVICE)

    # Distort images based on moving states from agent
    if agent is not None:
        agent_outputs = agent(words)
        S = torch.max(agent_outputs, 3).indices

        S2 = S.detach().clone().cpu()

        # Reverse direction of one fiducial point in each image
        rev_points = np.random.randint(S.shape[1], size=S.shape[0])
        mask = torch.zeros_like(S2, dtype=torch.bool)
        mask[torch.arange(rev_points.shape[0]), rev_points, :] = True
        S2 = torch.where(mask, 1 - S2, S2)

        # Augment
        aug_S, radii = distort_batch(images, words, bboxes, n_patches, radius, S.clone())
        aug_S2 = distort_batch(images, words, bboxes, n_patches, radius, S2.clone(), radii_list=radii)
        return aug_S, aug_S2, agent_outputs, S, S2
    
    # Random distortion
    else:
        S = np.random.randint(2,size=images.shape[0]*2*(n_patches+1)*2).reshape((images.shape[0], 2*(n_patches+1), 2))
        S = torch.from_numpy(S)

        aug_S, _ = distort_batch(images, words, bboxes, n_patches, radius, S)
        return aug_S

def learning_agent_loss(error, error_S2, agent_outputs, S, S2):
    """
    Loss function for the augmentation agent of Learn to Augment.
    @param error: character error rate of the recognizer on images augmented with moving state S
    @param error_S2: character error rate of the recognizer on images augmented with moving state S2
    @param agent_outputs: Learning agent outputs from this batch
    @param S: Distortion moving state
    @param S2: Random distortion moving state

    @return Learning agent loss.
    """
    S = S.to(DEVICE)
    S2 = S2.to(DEVICE)

    # Fiducial points where the random moving state reversed the direction
    rev_mask = S != S2

    # Transform agent outputs to probabilities
    agent_outputs = F.softmax(agent_outputs,3)

    # Compute P(S2|Iin) = S2_probs, and P(-S2|Iin) = S2_rev_probs
    S2_probs = torch.where(rev_mask, agent_outputs.min(-1).values, agent_outputs.max(-1).values)
    S2_rev_probs = 1 - S2_probs
    S2_probs = S2_probs.prod(-1)
    S2_rev_probs = S2_rev_probs.prod(-1)

    # If S2 increases difficulty learn from S2, and otherwise from reversed S2
    mask = error < error_S2
    mask = mask.unsqueeze(-1).expand_as(S2_probs).to(DEVICE)
    P = torch.where(mask, S2_probs, S2_rev_probs)
    loss = P.log().sum(-1) * -1
    loss = loss.mean()
    
    # if S2[0][0][0] == 0 and S2[0][0][1] == 0 and (S[0][0][0] != 0 or S[0][0][1] != 0):
    #     print(f"S2 Harder - S error {error[0]:.2f}, S2 error {error_S2[0]:.2f}, S {S[0][0].tolist()}, S2 {S2[0][0].tolist()}, loss {loss:.2f}, A1 {agent_outputs[0][0][0][0]:.2f}, A2 {agent_outputs[0][0][1][0]:.2f}")
    # elif S[0][0][0] == 0 and S[0][0][1] == 0 and (S2[0][0][0] != 0 or S2[0][0][1] != 0):
    #     print(f"S2 Easier - S error {error[0]:.2f}, S2 error {error_S2[0]:.2f}, S {S[0][0].tolist()}, S2 {S2[0][0].tolist()}, loss {loss:.2f}, A1 {agent_outputs[0][0][0][0]:.2f}, A2 {agent_outputs[0][0][1][0]:.2f}")
    # else:
    #     print(f"S {S[0][0].tolist()}, S2 {S2[0][0].tolist()}")
    #     print(f"S2 Similar - S {S[0][0].tolist()}, S2 {S2[0][0].tolist()}, loss {loss[0].item():.2f}, A1 {agent_outputs[0][0][0][0]:.2f}, A2 {agent_outputs[0][0][1][0]:.2f}")
    
    return loss

def train(error, error_S2, agent_opt, agent_outputs, S, S2):
    """
    Train the learning augmentation agent.
    @param error: character error rate of the recognizer on images augmented with moving state S
    @param error_S2: character error rate of the recognizer on images augmented with moving state S2
    @param agent_opt: Agent optimizer
    @param agent_outputs: Learning agent outputs from this batch
    @param S: Distortion moving state
    @param S2: Random distortion moving state

    @return Learning agent loss.
    """
    criterion = learning_agent_loss
    loss = criterion(error, error_S2, agent_outputs, S, S2)
    if not torch.isnan(loss):
        agent_opt.zero_grad()
        loss.backward()
        agent_opt.step()
        return loss.item()
    return 0.0

def line_to_words(image):
    """
    Segment a text image into images with individual words.
    @param image: Image to segment

    @return Word images and bounding box locations in the original image
    """
    _, binary = cv2.threshold(image, 128, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)

    # Dilation
    kernel = np.ones((20, 20), np.uint8)
    binary = cv2.dilate(binary, kernel, iterations=1)

    # Morphology
    kernel = np.ones((5, 5), np.uint8)
    binary = cv2.morphologyEx(binary, cv2.MORPH_CLOSE, kernel)

    # Find contours
    contours, _ = cv2.findContours(binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    # Remove bounding boxes that are too small
    min_width, min_height = 10, 10
    contours = [cnt for cnt in contours if cv2.boundingRect(cnt)[2] > min_width and cv2.boundingRect(cnt)[3] > min_height]

    # Show contours
    # image_with_contours = image.copy()
    # cv2.drawContours(image_with_contours, contours, -1, (0, 255, 0), 2)
    # plt.figure(figsize=(5, 5))
    # plt.imshow(cv2.cvtColor(image_with_contours, cv2.COLOR_BGR2RGB))
    # plt.title('Contours')
    # plt.axis('off')

    # Show bounding rectangles
    # image_with_rectangles = image.copy()
    # for cnt in contours:
    #     x, y, w, h = cv2.boundingRect(cnt)
    #     cv2.rectangle(image_with_rectangles, (x, y), (x + w, y + h), (0, 255, 0), 2)
    # plt.figure(figsize=(5, 5))
    # plt.imshow(cv2.cvtColor(image_with_rectangles, cv2.COLOR_BGR2RGB))
    # plt.title('Bounding Rectangles')
    # plt.axis('off')

    # Sort from left to right
    bounding_boxes = [cv2.boundingRect(cnt) for cnt in contours]
    bounding_boxes = sorted(bounding_boxes, key=lambda b: b[0])

    # Get word images
    word_images = []
    for bbox in bounding_boxes:
        x, y, w, h = bbox
        word_image = image[y:y+h, x:x+w]
        word_images.append(word_image)

    # Show word images
    # plt.figure(figsize=(5, 5))
    # for i, word_image in enumerate(word_images):
    #     plt.subplot(1, len(word_images), i + 1)
    #     plt.imshow(cv2.cvtColor(word_image, cv2.COLOR_BGR2RGB))
    #     plt.axis('off')
    # plt.show()

    return word_images, bounding_boxes

def augment_line_demo(line):
    """
    Demonstrate individual word augmentation by creating a gif of 100 different line augmentation (each time with only one word augmented).
    @param line: Image to augment
    """
    word_images, bounding_boxes = line_to_words(line)

    n_patches, radius = 3, 10
    distort_line_list = list()
    for _ in range(100):
        distort_line = line.copy()

        for word, bbox in zip(word_images, bounding_boxes):
            S = np.random.randint(2,size=2*(n_patches+1)*2).reshape((2*(n_patches+1), 2))
            word = cv2.resize(word, (100, 32))
            word, _ = distort(word, n_patches, radius, S)
            x, y, w, h = bbox
            word = cv2.resize(word, (w, h))
            distort_line[y:y+h, x:x+w] = word

        distort_line_list.append(distort_line)

    create_gif(distort_line_list, r'distort.gif')


if __name__ == '__main__':
    line = cv2.imread('img//a01-000u-00.png', cv2.IMREAD_GRAYSCALE)
    augment_line_demo(line)