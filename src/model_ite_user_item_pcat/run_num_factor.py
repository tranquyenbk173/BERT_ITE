import logging
from src.model_ite_user_item_pcat import model

# config log
logging.basicConfig(format='%(asctime)s : %(levelname)s : %(message)s', level=logging.INFO)


def test():
    save_path_name = 'test_both_embed_pcat'
    for data_name in ['recobell/']:
        if data_name == 'recobell/':
            epochs = 101
        else:
            epochs = 151
        for size in [8, 16, 32, 64]:
            logging.info("dataset: %s, num_factor: %s, save_path_name: %s", data_name, size, save_path_name)
            model.training_num_factors(size, data_name, save_path_name, epochs, save_model=False)


def main():
    save_path_name = 'model_ite_user_item_pcat'
    num_negatives = 4
    for data_name in ['tmall/']:
        if data_name == 'recobell/':
            epochs = 101
        else:
            epochs = 200
        for size in [64, 32, 16, 8]:
            logging.info("dataset: %s, num_factor: %s, save_path_name: %s", data_name, size, save_path_name)
            model.training_num_factors(size, data_name, save_path_name, epochs, num_negatives, save_model=True)


if __name__ == '__main__':
    main()
    # main()
