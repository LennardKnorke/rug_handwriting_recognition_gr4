Information on Structure and Usage of the DSS classifier pipeline.

Structure:
(CODE)
- main.py		Run with folder of images as argument e.g. "python main.py test/",
			will save an output txt file in .results/ for each input image.
- classification.py	Contains the code to train the classifier for segmented images
- augmentation.py	code for the augmentation (used in classification.py)
- network.py		Main CNN network defined (used in main.py and classification.py)
- utils.py		used through both training and final pipeline with general support and decoding functions
- ops.py		Contains augmentation operations (used in augmentation.py)
- segmentation.py	Contains code for image segmentation (used in main.py and classification.py)

(OUTCOMES)
- recognizer_N3_M1.pth	The best model (CNN) after training (parameters N=3 and M=1)
- encoder.joblib	Stores the labels of one-hot encoded characters

(Training prerequisite)
- ./img/segmented/	Folder with the character images provided for training

Info:
Training assumes the folder and files mentioned are in their current location.
For evaluation, make sure the working directory contains main.py (and all other listed files).
The output text files already correctly order the letters from right to left.
The number of lines may differ between the actual text due to inaccurate segmentation.