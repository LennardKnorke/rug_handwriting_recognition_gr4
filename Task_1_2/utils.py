import os

def get_image_paths(folder_path : str, mode : bool = True) -> list:
    """
    This function reads all the images in a folder and returns them as a list.
    @param folder_path: The path to the folder containing the images.
    @return list of images.
    """
    image_paths = []
    for file in os.listdir(folder_path):
        if file.endswith(".jpg"):
            file_type = file.split("-")[-1].split(".")[0]
            if (mode and file_type == "binarized") or (not mode and file_type == "R01"):
                image_paths.append(file)
    print("Found", len(image_paths), "images.")
    return image_paths