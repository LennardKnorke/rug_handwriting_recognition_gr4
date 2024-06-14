Information on Structure and Usage of the DSS classifier pipeline.

Structure:
(CODE)
- main.py		For evaluation, run with folder of images as argument e.g. "python main.py test/",
				will save an output txt file in .results/ for each input image.
				For training, run python main.py --help to see the available arguments.
				For example, to force training before evaluation, add '-f', and to create a 90/10 train/test split, add '-s 10'
				The default values for the parameters were deemed best.
- classification.py	Contains the code to train the classifier for segmented images
- augmentation.py	code for the augmentation (used in classification.py)
- learn_to_augment.py	code for the Learn to Augment augmentation (used in augmentation.py)
						Note that this is not used by default (instead RandomAugment is used)
- networks.py	Main CNN networks defined (used in main.py and classification.py)
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