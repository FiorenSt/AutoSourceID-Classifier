#
#                          ###           ######       ###      ### ##                    ######
#                         #####         ##            ###      ###  ###               ###
#                        ### ###        ##            ###      ###   ###            ###
#                       ###   ###         #####       ###      ###     ###    ###   ###
#                      ### ### ###            ##      ###      ###   ###            ###
#                     ###       ###           ##      ###      ###  ###               ###
#                    ###         ###    #######       ###      ### ##                    ######
#


from load_and_predict import load_and_predict
from prepare_data import prepare_data, load_test_data

def main(DATA_PATH='### Your data path ###',
         MODEL_NAME='model.h5',
         train_model=False):

    if train_model:
        print("Training the model...")
        train_dataset, valid_dataset, total_train, total_val = prepare_data(DATA_PATH)
        train_model(train_dataset, valid_dataset, total_train, total_val)
    else:
        print("Loading model and making predictions...")
        patches = load_test_data(DATA_PATH)
        predictions = load_and_predict(MODEL_NAME, patches)

        print(predictions)
if __name__ == "__main__":
    main()
