import os


TEXT_HEIGHT : int = 1000      # The height of the full unsegmented text
TEXT_WIDTH : int = 1000       # The width of the full unsegmented text

CHARACTER_HEIGHT : int = 100  # The height of a single character
CHARACTER_WIDTH : int = 100   # The width of a single character

USE_BINARY : bool = True       # Set to True if the images to use are already binarized. False if they are RGB and we binarize ourselves
BINARY_THRESHOLD : int = 100   # The threshold to use when binarizing the images.


def get_image_paths(folder_path : str) -> list:
    """
    This function reads all the images in a folder and returns them as a list.
    @param folder_path: The path to the folder containing the images.
    @return list of images.
    """
    image_paths = []
    for file in os.listdir(folder_path):
        if file.endswith(".jpg"):
            file_type = file.split("-")[-1].split(".")[0]
            if (USE_BINARY and file_type == "binarized") or (not USE_BINARY and file_type == "R01"):
                image_paths.append(folder_path+file)
    print("Found", len(image_paths), "images.")
    return image_paths