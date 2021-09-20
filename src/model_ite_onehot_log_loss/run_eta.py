import logging
from src.model_ite_onehot_log_loss import model

# config log
logging.basicConfig(format='%(asctime)s : %(levelname)s : %(message)s', level=logging.INFO)


def main():
    save_path_name = 'model_ite_onehot_log_loss'
    num_negatives = 4
    for data_name in ['tmall/']:
        if data_name == 'recobell/':
            epochs = 101
        else:
            epochs = 201
        for size in [0.65, 0.70, 0.75, 0.6]:
            logging.info("dataset: %s, eta: %s, save_path_name: %s", data_name, size, save_path_name)
            model.training_eta(size, data_name, save_path_name, epochs, num_negatives, save_model=False)


if __name__ == '__main__':
    main()
