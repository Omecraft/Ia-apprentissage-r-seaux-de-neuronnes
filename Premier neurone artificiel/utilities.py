import h5py
import numpy as np
import os


def load_data():
	base_dir = os.path.dirname(__file__)
	train_path = os.path.join(base_dir, 'datasets', 'trainset.hdf5')
	test_path = os.path.join(base_dir, 'datasets', 'testset.hdf5')

	if not os.path.isfile(train_path):
		raise FileNotFoundError(f"Expected training dataset at: {train_path}")
	if not os.path.isfile(test_path):
		raise FileNotFoundError(f"Expected test dataset at: {test_path}")

	with h5py.File(train_path, "r") as train_dataset, h5py.File(test_path, "r") as test_dataset:
		X_train = np.array(train_dataset["X_train"][:])  # train features
		y_train = np.array(train_dataset["Y_train"][:])  # train labels
		X_test = np.array(test_dataset["X_test"][:])    # test features
		y_test = np.array(test_dataset["Y_test"][:])    # test labels

	return X_train, y_train, X_test, y_test