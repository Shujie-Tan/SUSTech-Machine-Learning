# Import the functions to calculate feature descriptors
from skimage.feature import local_binary_pattern
from skimage.feature import hog
from skimage.io import imread
from sklearn.externals import joblib
# To read file names
import argparse as ap
import glob
import os
import config

if __name__ == "__main__":
    # Argument Parser
    parser = ap.ArgumentParser()
    parser.add_argument('-p', "--pospath", help="Path to positive images",
                        required=False)
    parser.add_argument('-n', "--negpath", help="Path to negative images",
                        required=False)
    parser.add_argument('-d', "--descriptor", help="Descriptor to be used -- HOG",
                        default="HOG")
    args = vars(parser.parse_args())

    pos_im_path = args["pospath"]  # "./CarData/TrainImages/"
    neg_im_path = args["negpath"]  # "./CarData/TrainImages/"

    des_type = args["descriptor"]

    # If feature directories don't exist, create them
    if not os.path.isdir(config.pos_feat_ph):
        os.makedirs(config.pos_feat_ph)

    # If feature directories don't exist, create them
    if not os.path.isdir(config.neg_feat_ph):
        os.makedirs(config.neg_feat_ph)

    print("Calculating the descriptors for the positive samples and saving them")
    for im_path in glob.glob(os.path.join(pos_im_path, "*")):
        im = imread(im_path, as_grey=True)
        if des_type == "HOG":
            fd = hog(im, config.orientations,
                     config.pixels_per_cell, config.cells_per_block)
        fd_name = os.path.split(im_path)[1].split(".")[0] + ".feat"
        fd_path = os.path.join(config.pos_feat_ph, fd_name)
        joblib.dump(fd, fd_path)
    print("Positive features saved in {}".format(config.pos_feat_ph))

    print("Calculating the descriptors for the negative samples and saving them")
    for im_path in glob.glob(os.path.join(neg_im_path, "*")):
        im = imread(im_path, as_grey=True)
        if des_type == "HOG":
            fd = hog(im, config.orientations,
                     config.pixels_per_cell, config.cells_per_block)
        fd_name = os.path.split(im_path)[1].split(".")[0] + ".feat"
        fd_path = os.path.join(config.neg_feat_ph, fd_name)
        joblib.dump(fd, fd_path)
    print("Negative features saved in {}".format(config.neg_feat_ph))

    print("Completed calculating features from training images")
