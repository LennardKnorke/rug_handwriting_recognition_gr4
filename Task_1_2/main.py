import os
import argparse
import shutil

from segmentation import extract_and_resize_characters
from classification import *
from utils import *

def parse_args() -> argparse.Namespace:
    """Parse the command-line arguments"""

    parser = argparse.ArgumentParser(description="")

    parser.add_argument(
        '--input_path',
        type=str,
        default=os.path.join("img", "test"),
        help='Path to the folder containing the test images.'
    )

    parser.add_argument(
        '--output_path',
        type=str,
        default="test_evaluation",
        help='Path to the folder where the output txt files should be placed.'
    )

    parser.add_argument(
        '-a',
        '--augment',
        type=str,
        choices=['elastic', 'randaug'],
        nargs='*',
        default=['randaug'],
        help='Augmentation method(s) to use while training the classifier: Elastic morphing and/or RandomAugment.'
    )

    parser.add_argument(
        '--lta',
        action=argparse.BooleanOptionalAction,
        help='Train an elastic morphing policy using Learn to Augment.'
    )

    parser.add_argument(
        '-f',
        '--force-train',
        action=argparse.BooleanOptionalAction,
        help='Train the classifier even if it already exists.'
    )

    parser.add_argument(
        '-e',
        '--epochs',
        type=int,
        default=160,
        help='Number of epochs to train the classifier for.'
    )

    parser.add_argument(
        '-b',
        '--batch-size',
        type=int,
        default=64,
        help='Batch size for training the classifier.'
    )

    parser.add_argument(
        '-p',
        '--patches',
        type=int,
        default=2,
        help='Number of patches to use for Learn to Augment.'
    )

    parser.add_argument(
        '-r',
        '--radius',
        type=int,
        default=10,
        help='Radius to use for Learn to Augment.'
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
        '-s',
        '--test-split',
        type=int,
        default=0,
        choices=range(100),
        help='Percentage of test pre-segmented characters. Set to 0 to disable.'
    )

    args = parser.parse_args()

    if args.lta and "elastic" not in args.augment:
        args.augment.append("elastic")

    if "elastic" not in args.augment:
        args.patches, args.radius = None, None
    if "randaug" not in args.augment:
        args.n_randaug, args.m_randaug = None, None

    if not os.path.exists(args.input_path):
        parser.error('The input folder does not exist.')

    return args


def run_DSS() -> None:
    """
    Loads the best saved model and runs it on new DSS data available in filepath. 
    Expects path to be filled with images png type 
    Will print results and save prediction in a file
    """
    args = parse_args()

    # Split train/test
    if args.test_split and not os.path.exists(get_test_split_filename(args.test_split)):
        create_test_split(args.test_split)

    # Train classifier
    classifier_name = uniquify(get_model_save_name('recognizer', args.patches, args.radius, args.n_randaug, args.m_randaug), find=True)
    if args.force_train or not os.path.exists(os.path.join("models", f"{classifier_name}.pth")):
        train(args.epochs,
              args.batch_size,
              args.patches,
              args.radius,
              args.n_randaug,
              args.m_randaug,
              elastic="elastic" in args.augment,
              lta=args.lta,
              train_recog=True,
              randaug="randaug" in args.augment,
              split=args.test_split
              )
              

    # Test classifier on pre-segmented test set
    if args.test_split:
        load_and_test(args.batch_size, classifier_name, args.test_split)

    # Load classifier and encoder
    classifier = load_classifier(classifier_name)
    encoder = load("encoder.joblib")
    classes = encoder.categories_[0]

    # Create output folder
    os.makedirs(args.output_path, exist_ok=True)

    # Segment test images
    for filename in os.listdir(args.input_path):
        # Get filename of binarized image
        f, ext = os.path.splitext(filename)
        if ext != '.pbm' and "binarized" not in f: continue
        file = os.path.join(args.input_path, filename)

        # Create directory for segmented characters
        segmented_dir = os.path.join("img", "test_segmented", f)
        if os.path.isdir(segmented_dir):
            shutil.rmtree(segmented_dir)
        os.makedirs(segmented_dir)

        # Segment
        segmented_characters = extract_and_resize_characters(file)
        transcript = []
        for line_id, line in enumerate(segmented_characters):
            transcript.append("")
            for char_id, img in enumerate(line):
                # Save segmented characters by index: "{line}-{char}.png"
                name = f'{line_id}-{char_id}.png'
                cv2.imwrite(os.path.join(segmented_dir, name), img)

                # Classify character and add to transcript
                img = torch.from_numpy(img.reshape((1, 1, CHARACTER_HEIGHT, CHARACTER_WIDTH))).float().to(device)
                output = classifier(img)
                letter = hebrewize(encoder.inverse_transform(output.detach().cpu().numpy())[0][0].decode())
                transcript[-1] += letter

            # Reverse characters in line
            transcript[-1] = transcript[-1][::-1]

        # Write transcript to file
        transcript = "\n".join(transcript)
        with open(os.path.join(args.output_path, f"{f}.txt"), "w", encoding='utf-8') as text_file:
            text_file.write(transcript)

        

        

##############################################
# Main script. This script will run the entire pipeline.
##############################################
if __name__ == '__main__':
    print("RUNNING TASK 1. THE DEAD SEA SCROLLS")
    run_DSS()
    