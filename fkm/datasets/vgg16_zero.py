"""

    https://www.kaggle.com/code/viratkothari/image-classification-of-mnist-using-vgg16/notebook
    Image Classification of MNIST using VGG16

    https://marubon-ds.blogspot.com/2017/09/vgg-fine-tuning-model.html
    Fine tune VGG

"""
import json
import os
import pickle

import PIL.Image
import keras
import matplotlib.pyplot as plt
import numpy as np
import sklearn
import tensorflow as tf
from keras import layers
# Library for Transfer Learning
from keras.applications import VGG16, VGG19, NASNetLarge, ResNet50
# from keras.applications.vgg16 import preprocess_input
# from keras.preprocessing import image
# Libraries for TensorFlow
from keras.utils import to_categorical
from sklearn.model_selection import train_test_split

H = 32
W = 32
OVERWRITE = False
OVERWRITE_DATA = False
print('gpu_name: ', tf.test.gpu_device_name(), ', is_gpu: ', tf.test.is_gpu_available())

def show_img(X, name='0', root_dir='.'):
	# img = PIL.Image.fromarray(X)
	# plt.imshow(img.astype('int32'))
	plt.imshow(X)
	if not os.path.exists(root_dir):
		os.makedirs(root_dir)
	plt.savefig(os.path.join(root_dir, f'{name}.png'))


# plt.show()

#
# def predict(model, img_name):
# 	img = image.load_img(img_name, target_size=(48, 48))
# 	img = image.img_to_array(img)
# 	plt.imshow(img.astype('int32'))
# 	# plt.show()
# 	img = preprocess_input(img)
#
# 	prediction = model.predict(img.reshape(1, 48, 48, 3))
# 	output = np.argmax(prediction)
#

# print(class_names[output] + ": " + Get_Element_Name(class_names[output]))


def VGG16_MODEL(n_classes=62, out_dim = 100):
	print('Load VGG16')
	# VGG16 only works for 3 channels (input layer)
	# let us prepare our input_layer to pass our image size. default is (224,224,3). we will change it to (32,32,3)
	input_layer = layers.Input(shape=(H, W, 3))  # the minimal size is 32 x 32 for VGG16
	# initialize the transfer model VGG16 with appropriate properties per our need.
	# we are passing parameters as following
	# 1) weights='imagenet' - Using this we are carring weights as of original weights.
	# 2) input_tensor to pass the VGG16 using input_tensor
	# 3) we want to change the last layer so we are not including top layer
	model_vgg16 = VGG16(weights='imagenet', input_tensor=input_layer, include_top=False)
	# model_vgg16 = VGG19(weights='imagenet', input_tensor=input_layer, include_top=False)
	# model_vgg16 = ResNet50(weights='imagenet', input_tensor=input_layer, include_top=False)
	# model_vgg16 = NASNetLarge(weights='imagenet', input_tensor=input_layer, include_top=False)
	# See the summary of the model with our properties.
	# model_vgg16.summary()
	# print("Summary of Custom VGG16 model.\n")
	# print("1) We flatten the last layer and added 1 Dense layer and 1 output layer.\n")
	last_layer = model_vgg16.output  # we are taking last layer of the model

	# Add flatten layer: we are extending Neural Network by adding flatten layer
	flatten = layers.Flatten()(last_layer)
	# # Add dense layer
	# dense1 = layers.Dense(100, activation='relu')(flatten)
	# dense1 = layers.Dense(100, activation='relu')(flatten)
	# dense1 = layers.Dense(100, activation='relu')(flatten)

	# Add dense layer to the final output layer
	flatten = layers.Dense(out_dim, activation='relu')(flatten)
	# flatten = layers.Dense(300, activation='relu')(flatten)
	output_layer = layers.Dense(n_classes, activation='softmax')(flatten)
	# Creating model with input and output layer
	model = keras.models.Model(inputs=input_layer, outputs=output_layer)
	# Summarize the model
	# model.summary()

	# we will freeze all the layers except the last layer
	# we are making all the layers intrainable except the last layer
	# print("We are making all the layers intrainable except the last layer. \n")
	# for layer in model.layers[:-1]:
	# 	layer.trainable = False
	# # model.summary()

	# Compiling Model
	model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
	print("Model compilation completed.")
	model.summary()

	return model


def resize_img(X):
	X = X.reshape(-1, 28, 28)
	show_img(X[0, :, :])  # for debug
	X = np.repeat(X[:, :, :, np.newaxis], 3, axis=3)  # duplicate 3 channels
	# show_img(X[0, :, :, :])
	# Resize the images 32*32 as required by VGG16
	try:
		from keras.preprocessing.image import img_to_array, array_to_img
	except Exception as e:
		from tensorflow.keras.utils import img_to_array, array_to_img
	# here * 255 is very important. If you don't do it, after resize, the image will be weired.
	# VGG needs 32 * 32
	X = np.asarray(
		[img_to_array(array_to_img(im * 255, scale=False).resize((H, W), resample=PIL.Image.LANCZOS)) / 255 for im in
		 X])
	# X1 = np.asarray([np.array(PIL.Image.fromarray(im, mode='RGB').resize((H,W), resample=PIL.Image.LANCZOS)) for im in X])

	return X


def get_all_Xy(data_dir='data/all_data_raw'):
	# Train the Model
	X = []
	y = []
	for i, json_file in enumerate(os.listdir(data_dir)):
		with open(os.path.join(data_dir, json_file), 'rb') as f:
			vs = json.load(f)
		print(i, json_file, len(vs['users']))
		for user in vs['users']:
			tmp = vs['user_data'][user]
			xtrain, ytrain = tmp['x'], tmp['y']
			X.extend(xtrain)
			y.extend(ytrain)
		# 	break
		# break
	X = np.asarray(X)
	y = np.asarray(y)

	print("Resize the data to HXW (32x32) for VGG16")
	X = resize_img(X)

	return X, y


def plot_training(history, root_dir='.'):
	# plot the loss and accuracy
	try:
		acc = history.history['acc']
		val_acc = history.history['val_acc']
	except Exception as e:
		acc = history.history['accuracy']
		val_acc = history.history['val_accuracy']
	loss = history.history['loss']
	val_loss = history.history['val_loss']
	epochs = range(1, len(acc) + 1)

	plt.title('Training and validation accuracy')
	plt.plot(epochs, acc, '-*', color='b', label='Training acc')
	plt.plot(epochs, val_acc, '-o', color='r', label='Validation acc')
	plt.legend()
	plt.savefig(os.path.join(root_dir, 'acc.png'))
	# plt.show()

	plt.figure()
	plt.title('Training and validation loss')
	plt.plot(epochs, loss, '-*', color='b', label='Training loss')
	plt.plot(epochs, val_loss, '-o', color='r', label='Validation loss')
	plt.legend()
	plt.savefig(os.path.join(root_dir, 'loss.png'))


# plt.show()


def report(model, xtest, ytest):
	# This function helps to predict individual image supplied to it
	prediction = model.predict(xtest)
	y_pred = np.argmax(prediction, axis=1)
	# print(y_pred)

	ytest_true = np.argmax(ytest, axis=1)

	report = sklearn.metrics.classification_report(ytest_true, y_pred)
	# print(report)

	# not support for multiclass
	# fpr, tpr, thresholds = sklearn.metrics.roc_curve(ytest_true, np.max(prediction, axis=1))
	# auc = sklearn.metrics.auc(fpr, tpr)
	# print(auc)

	# cm = sklearn.metrics.confusion_matrix(ytest_true, y_pred)
	# print(cm)

	acc = sklearn.metrics.accuracy_score(ytest_true, y_pred)
	print(f'accuracy: {acc}')


def load_model(model_file, ROOT_DIR='.'):
	if OVERWRITE and os.path.exists(model_file): os.remove(model_file)
	if not os.path.exists(model_file):
		print('Loading data.')
		data_file  = f'{ROOT_DIR}/data/xy.dat'
		if OVERWRITE_DATA and os.path.exists(data_file): os.remove(data_file)
		if not os.path.exists(data_file):
			X, y = get_all_Xy(data_dir=f'{ROOT_DIR}/data/all_data_raw')
			with open(data_file, 'wb') as f:
				pickle.dump((X, y), f,  protocol=4)
		else:
			with open(data_file, 'rb') as f:
				X, y = pickle.load(f)
		y_onehot = to_categorical(y)

		# split the data into train and test
		print("Splitting data for train and test.")
		xtrain, xtest, ytrain, ytest = train_test_split(X, y_onehot, test_size=0.01, random_state=42)
		print(xtrain.shape, xtest.shape, ytrain.shape, ytest.shape)

		print('Fine-tuning VGG16 on FEMNIST')
		model = VGG16_MODEL(n_classes=62)
		# with tensorflow.device('/gpu:0'): # Keras default uses gpu if it's available
		print('gpu_name: ', tf.test.gpu_device_name(), ', is_gpu: ', tf.test.is_gpu_available())
		history = model.fit(xtrain, ytrain, epochs=10, batch_size=128, verbose=True, validation_data=(xtest, ytest))
		print("Fitting the model completed.")

		print('\nPlotting training results.')
		plot_training(history, ROOT_DIR)

		print('\nTraining report.')
		report(model, xtrain, ytrain)
		print('\nTesting report.')
		report(model, xtest, ytest)

		# dump model to disk
		model.save(model_file)
	# del model  # deletes the existing model
	else:
		# Reload the model
		model = keras.models.load_model(model_file)
	return model


def gen_cnn_features(model, X, batch_size=128):
	# from fkm.datasets.vgg16_zero import resize_img, show_img, VGG16_MODEL, plot_training, report
	X = resize_img(X)
	show_img(X[0, :, :, :], 'tmp.png')

	from keras import backend as K
	# https://stackoverflow.com/questions/41711190/keras-how-to-get-the-output-of-each-layer
	get_2rd_layer_output = K.function([model.layers[0].input],
	                                  [model.layers[-2].output])
	N = X.shape[0]
	X_cnn = []
	for i in range(0, N, batch_size):
		tmp = X[i:i+batch_size]
		layer_output = get_2rd_layer_output([tmp])[0]
		X_cnn.extend(layer_output)

	return np.asarray(X_cnn)

def main():
	"""
		ONLY work for FEMNIST
	:return:
	"""
	DATASET_NAME = 'femnist'
	ROOT_DIR = '.'
	model_file = os.path.join(ROOT_DIR, f'vgg16_{DATASET_NAME}_zero.h5')

	print(f'load/generate model_file: {model_file}')
	model = load_model(model_file, ROOT_DIR)
	# Summarize the model
	model.summary()

	print('generate cnn features')
	gen_cnn_features(ROOT_DIR, model)

	print('finished.')


if __name__ == "__main__":
	main()
