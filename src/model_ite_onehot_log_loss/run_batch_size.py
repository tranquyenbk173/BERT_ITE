import logging
from src.model_ite_onehot_log_loss import model

# config log
logging.basicConfig(format='%(asctime)s : %(levelname)s : %(message)s', level=logging.INFO)


def main():
    save_path_name = 'model_ite_onehot_log_loss'
    num_negatives = 4
    for data_name in ['ml-1m/']:
        if data_name == 'recobell/':
            epochs = 101
        else:
            epochs = 151
        for size in [512, 1024, 2048, 4096]:
            logging.info("dataset: %s, batch_size: %s, save_path_name: %s", data_name, size, save_path_name)
            model.training_batch_size(size, data_name, save_path_name, epochs, num_negatives, save_model=False)


if __name__ == '__main__':
    main()
