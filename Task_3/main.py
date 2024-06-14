import cv2
import os
import torch
import sys

from network import Recurrent_CNN
from utils import IMAGE_HEIGHT, IMAGE_WIDTH, N_CHARS, DEVICE, ctc_decode, resize_and_pad


def set_up_image(image_path : str) -> torch.Tensor:
    """
    Given a path to an image, this function reads the image, resizes it, normalizes it and converts it to a tensor
    @param image_path: path to the image
    @return: torch tensor of the image
    """
    img = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
    img = resize_and_pad(img, (IMAGE_WIDTH, IMAGE_HEIGHT))
    img = img / 255.0
    img = torch.from_numpy(img).float()
    img = img.unsqueeze(2)
    img = img.permute(2, 0, 1).unsqueeze(0)
    return img

if __name__ == "__main__":
    # Read Arguments (Path to image data)
    if len(sys.argv) < 2:
        print("Please provide the path to the dataset")
        sys.exit(0)

    image_dir : str = sys.argv[1]

    if not os.path.exists(image_dir):
        print("The provided path does not exist")
        sys.exit(0)
    
    results_dir : str = "results/"
    if not os.path.exists(results_dir):
        os.makedirs(results_dir)
    

    # Load Model
    model = Recurrent_CNN(num_classes = N_CHARS + 1).requires_grad_(False)
    model.load_state_dict(torch.load("IAM_the_best_model.pth"))
    model = model.to(DEVICE)


    # Get image paths and names without extension
    image_names = [os.path.basename(f).split('.')[0] for f in os.listdir(image_dir) if f.endswith(".png")]
    image_paths = [os.path.join(image_dir, img_name)+".png" for img_name in image_names]

    # Run Model on Testing Images
    for img_path, name in zip(image_paths, image_names):
        img = set_up_image(img_path)
        img = img.to(DEVICE)

        # Get output
        output_rnn, output_ctc = model(img)
        pred_ints = output_rnn.argmax(2).cpu().numpy()
        decoded_str = ctc_decode(pred_ints)[0]
        decoded_str = decoded_str.strip()
        print(decoded_str)

        # Save output
        output_path = os.path.join(results_dir, name)
        with open(output_path+".txt", "w") as f:
            f.write(decoded_str)
            f.close()




    
