from model import random_forest_model, kernel_ridge_regression, gaussian_process_regression, load_model, load_keras_model
import sys

if (sys.argv[1] == 'train'):
    if (sys.argv[2] == "forest"):
        random_forest_model(int(sys.argv[3]), int(sys.argv[4]), sys.argv[5], sys.argv[6])
    elif (sys.argv[2] == "ridge_kernel"):
        kernel_ridge_regression(sys.argv[3], float(sys.argv[4]), sys.argv[5], sys.argv[6])
    elif (sys.argv[2] == "gaussian_process"):
        gaussian_process_regression(sys.argv[3], float(sys.argv[4]), sys.argv[5], sys.argv[6])
    elif (sys.argv[2] == "enn"):
        ensemble_neural_network(int(sys.argv[3]), sys.argv[4], sys.argv[5])
    else:
        print("Invalid input")
elif (sys.argv[1] == "test"):
    if sys.argv[2][-2:] == "h5":
        load_keras_model(sys.argv[2], sys.argv[3])
    else:
        load_model(sys.argv[2], sys.argv[3])
else:
    print("Invalid input")
