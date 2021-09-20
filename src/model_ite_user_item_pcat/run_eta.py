import logging
from src.model_ite_user_item_pcat import model

# config log
logging.basicConfig(format='%(asctime)s : %(levelname)s : %(message)s', level=logging.INFO)


def main():
    num_negatives = 4
    save_path_name = 'model_ite_user_item_pcat'
    for data_name in ['ml-1m/']:
        if data_name == 'recobell/':
            epochs = 101
        else:
            epochs = 151
        for size in [0.1, 0.5, 1.0, 2.0]:
            logging.info("dataset: %s, eta: %s, save_path_name: %s", data_name, size, save_path_name)
            model.training_eta(size, data_name, save_path_name, epochs, num_negatives, save_model=False)


if __name__ == '__main__':
    main()
