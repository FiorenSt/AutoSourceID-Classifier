#
#                          ###           ######       ###      ### ##                   ######
#                         #####         ##            ###      ###  ###              ###
#                        ### ###        ##            ###      ###   ###           ###
#                       ###   ###         #####       ###      ###     ###    ###  ###
#                      ### ### ###            ##      ###      ###   ###           ###
#                     ###       ###           ##      ###      ###  ###              ###
#                    ###         ###    #######       ###      ### ##                   ######
#

############# TEMPORARY #########################
import os

import numpy as np

os.chdir('C:/Users/fiore/Desktop/UNI/Projects/Project8-FeatureExtraction/Flux Regression/ASID_FE/src/')
############# TEMPORARY #########################


from load_and_predict import load_and_predict
from prepare_data import prepare_data



# Now any file paths will be relative to the new working directory



def main(DATA_PATH='../Data/ML1_20200101_185651_GaiaDR3.fits',
         COORD_PATH='../Data/coordinates.csv',
         MODEL_NAME='model.h5',
         train_model=False,
         demo_plot=False,
         epochs=10,
         hdu=0):

    # ---------------------
    # Method choice
    # ---------------------
    if train_model:

        # ---------------------
        # Train the Part I model
        # ---------------------

        print("Training Part I Model...")
        train_dataset, valid_dataset, total_train, total_val = prepare_data(DATA_PATH)

        # Train the model
        train_model(train_dataset, valid_dataset, total_train, total_val)

        exit()
    else:
        # ---------------------
        # Load model and predict on synthetic images
        # ---------------------
        patches = prepare_data(DATA_PATH, COORD_PATH)
        print("Loading model and predicting on synthetic images...")

        predictions = load_and_predict(MODEL_NAME, np.array(patches))
        print(predictions)


if __name__ == "__main__":
    main()