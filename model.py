from joblib import dump, load
from sklearn.ensemble import RandomForestRegressor
from sklearn.kernel_ridge import KernelRidge
from sklearn.gaussian_process import GaussianProcessRegressor
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
from itertools import chain, combinations
from sklearn.gaussian_process.kernels import RationalQuadratic, RBF, DotProduct, WhiteKernel, Matern
from sklearn.metrics import mean_absolute_error, mean_absolute_percentage_error, mean_squared_error
import tensorflow as tf
import os
from preprocessed import data_preprocess

def random_forest_model(num_trees=400, max_depth=20, model_name = 'rf', data_dir = 'VN_housing_dataset.csv'):
    model = RandomForestRegressor(n_estimators=num_trees, max_depth=max_depth, min_samples_leaf=2, max_features='sqrt')
    X_train, X_test, y_train, y_test = data_preprocess(data_dir)
    model.fit(X_train, y_train)
    print("Model MAE on training set", mean_absolute_error(y_train, model.predict(X_train)))
    print("Model MAE on test set", mean_absolute_error(y_test, model.predict(X_test)))
    print("Model MSE on training set", mean_squared_error(y_train, model.predict(X_train)))
    print("Model MSE on test set", mean_squared_error(y_test, model.predict(X_test)))
    print("Model MAPE on training set", mean_absolute_percentage_error(y_train, model.predict(X_train)))
    print("Model MAPE on test set", mean_absolute_percentage_error(y_test, model.predict(X_test)))
    dump(model, os.path.join('saved_model', model_name + '.joblib'))
    print("Saved the model at " + os.path.join('saved_model', model_name + '.joblib'))

def kernel_ridge_regression(kernel='linear', alpha=1, model_name = 'kr', data_dir = 'VN_housing_dataset.csv'):
    model = KernelRidge(alpha=alpha, kernel=kernel, gamma=0.1)
    X_train, X_test, y_train, y_test = data_preprocess(data_dir)
    model.fit(X_train, y_train)
    print("Model MAE on training set", mean_absolute_error(y_train, model.predict(X_train)))
    print("Model MAE on test set", mean_absolute_error(y_test, model.predict(X_test)))
    print("Model MSE on training set", mean_squared_error(y_train, model.predict(X_train)))
    print("Model MSE on test set", mean_squared_error(y_test, model.predict(X_test)))
    print("Model MAPE on training set", mean_absolute_percentage_error(y_train, model.predict(X_train)))
    print("Model MAPE on test set", mean_absolute_percentage_error(y_test, model.predict(X_test)))
    dump(model, os.path.join('saved_model', model_name + '.joblib'))
    print("Saved the model at " + os.path.join('saved_model', model_name + '.joblib'))

def gaussian_process_regression(kernel='Matern', alpha=1, model_name = 'gp', data_dir = 'VN_housing_dataset.csv'):
    if kernel == 'RBF':
        model = GaussianProcessRegressor(kernel = RBF() + WhiteKernel(alpha), normalize_y = True)
    elif kernel == 'Matern':
        model = GaussianProcessRegressor(kernel = Matern() + WhiteKernel(alpha), normalize_y = True)
    elif kernel == "DotProduct":
        model = GaussianProcessRegressor(kernel = DotProduct() + WhiteKernel(alpha), normalize_y = True)
    elif kernel == "RationalQuadratic":
        model = GaussianProcessRegressor(kernel = RationalQuadratic() + WhiteKernel(alpha), normalize_y = True)
    else:
        print("Invalid kernel")
        return
    X_train, X_test, y_train, y_test = data_preprocess(data_dir)
    model.fit(X_train, y_train)
    print("Model MAE on training set", mean_absolute_error(y_train, model.predict(X_train)))
    print("Model MAE on test set", mean_absolute_error(y_test, model.predict(X_test)))
    print("Model MSE on training set", mean_squared_error(y_train, model.predict(X_train)))
    print("Model MSE on test set", mean_squared_error(y_test, model.predict(X_test)))
    print("Model MAPE on training set", mean_absolute_percentage_error(y_train, model.predict(X_train)))
    print("Model MAPE on test set", mean_absolute_percentage_error(y_test, model.predict(X_test)))
    dump(model, os.path.join('saved_model', model_name + '.joblib'))
    print("Saved the model at " + os.path.join('saved_model', model_name + '.joblib'))

def ensemble_neural_network(neurons=25, model_name = 'enn', data_dir = 'VN_housing_dataset.csv'):
    X_train, X_test, y_train, y_test = data_preprocess(data_dir)
    X_train = tf.convert_to_tensor(X_train, dtype=tf.float64)
    X_test = tf.convert_to_tensor(X_test, dtype=tf.float64)
    y_train = tf.convert_to_tensor(y_train, dtype=tf.float64)
    y_test = tf.convert_to_tensor(y_test, dtype=tf.float64)
    list_of_layers =[]
    Layer_0 = Dense(units=neurons, input_dim=X_train.shape[1], kernel_initializer='normal', activation='relu')
    for i in range(3):
      list_of_layers.append(Dense(units=neurons, kernel_initializer = 'normal', activation = 'relu'))
    final_layer = Dense(1, kernel_initializer='normal')
    def powerset(iterable):
        "powerset([1,2,3]) → () (1,) (2,) (3,) (1,2) (1,3) (2,3) (1,2,3)"
        s = list(iterable)
        return chain.from_iterable(combinations(s, r) for r in range(len(s)+1))
    for x in powerset(list_of_layers):
        temp_list_of_layers =list(x)
        ANN_temp = Sequential()

        ANN_temp.add(Layer_0)
        for layer in temp_list_of_layers:
            ANN_temp.add(layer)

        ANN_temp.add(final_layer)
        ANN_temp.compile(loss='mean_squared_error', optimizer='adam')
        ANN_temp.fit(x = X_train, y = y_train, batch_size = X_train.shape[0], epochs=3000, verbose=0)
    print("Model MAE on training set", mean_absolute_error(y_train.numpy(), ANN_temp.predict(X_train)))
    print("Model MAE on test set", mean_absolute_error(y_test.numpy(), ANN_temp.predict(X_test)))
    print("Model MSE on training set", mean_squared_error(y_train.numpy(), ANN_temp.predict(X_train)))
    print("Model MSE on test set", mean_squared_error(y_test.numpy(), ANN_temp.predict(X_test)))
    print("Model MAPE on training set", mean_absolute_percentage_error(y_train.numpy(), ANN_temp.predict(X_train)))
    print("Model MAPE on test set", mean_absolute_percentage_error(y_test.numpy(), ANN_temp.predict(X_test)))
    ANN_temp.save(os.path.join('saved_model', model_name + '.h5'))

def load_model(model_dir, input_data_dir="VN_housing_dataset.csv"):
    model = load(model_dir)
    X, y = data_preprocess(input_data_dir, train_test=False)
    print("Model MAE on test set", mean_absolute_error(y, model.predict(X)))
    print("Model MSE on test set", mean_squared_error(y, model.predict(X)))
    print("Model MAPE on test set", mean_absolute_percentage_error(y, model.predict(X)))
    print("The prediction output:")
    for i in list(model.predict(X)):
        print(i)

def load_keras_model(model_dir, input_data_dir="VN_housing_dataset.csv"):
    model = tf.keras.models.load_model(model_dir)
    X, y = data_preprocess(input_data_dir, train_test=False)
    X = tf.convert_to_tensor(X, dtype=tf.float64)
    y = tf.convert_to_tensor(y, dtype=tf.float64)
    print("Model MAE on test set", mean_absolute_error(y.numpy(), model.predict(X)))
    print("Model MSE on test set", mean_squared_error(y.numpy(), model.predict(X)))
    print("Model MAPE on test set", mean_absolute_percentage_error(y.numpy(), model.predict(X)))
    print("The prediction output:")
    for i in list(model.predict(X)):
        print(i)
