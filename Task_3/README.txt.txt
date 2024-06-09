Information on Structure and Usage of the IAM classifier pipeline.

Structure:
(CODE)
- main.py		Run with folder of images as argument e.g. "python main.py text/"
- training.py		Runs the complete training of a network, saving the best model and graph with results

- augmentation.py	code for the augmentation (used in training.py)
- dataset.py		loading the IAM dataset during trianing  (used only in training.py)
- network.py		Main CRNN network defined, as well as the cnn used during augmentation (used in main.py and training.py)
- utils.py		used through both training and final pipeline with general support and decoding functions

(OUTCOMES)
- IAM_the_best_model.pth	The best model after training
- 3 Plots with performance	(cer_plot.png, wer_plot.png, loss_plot.png)

(Training prerequisite)
- ./img/		Folder with the images provided for training
- img_lines_gt.txt	Text file with ground truths to the IAM dataset in "img/"

Info:
Training assumes the folders and files mentioned are in their current location.