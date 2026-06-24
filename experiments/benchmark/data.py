import numpy as np
import imblearn.datasets as imb_datasets
from sklearn.model_selection import train_test_split


def load_datasets(dataset_names, test_size=0.2, random_state=42):
    train_test_list = []
    for data_name in dataset_names:
        try:
            dataset = imb_datasets.fetch_datasets()[data_name]
            X, y = dataset["data"], dataset["target"]

            X_train, X_test, y_train, y_test = train_test_split(
                X, y, test_size=test_size, random_state=random_state, stratify=y
            )

            y_train = np.where(y_train == -1, 0, y_train)
            y_test = np.where(y_test == -1, 0, y_test)

            train_test_list.append((data_name, (X_train, X_test, y_train, y_test)))

            pos = np.sum(y == 1)
            print(f"Loaded {data_name}")
            print(f"\t pos {pos}")
            print(f"\t neg {len(y) - pos}")
        except Exception as e:
            print(f"Could not load {data_name}: {e}")

    return train_test_list
