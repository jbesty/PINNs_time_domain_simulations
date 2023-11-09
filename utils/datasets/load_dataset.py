import pickle

from definitions import PROJECT_PATH


def load_dataset_by_name(dataset_name):
    dataset_path = PROJECT_PATH / "utils" / "datasets" / f"{dataset_name}.pickle"

    with open(dataset_path, "rb") as file_path:
        dataset = pickle.load(file_path)

    return dataset
