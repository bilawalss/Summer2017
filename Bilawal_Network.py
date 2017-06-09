from __future__ import division

import keras
from keras.preprocessing.image import ImageDataGenerator
from keras.models  import Sequential
from keras.layers import *
from keras import backend as K
from scipy.misc import imread
import sys, os
from sklearn.preprocessing import LabelEncoder


###### preprocessing image #######




# augmentation configuration for training
train_datagen = ImageDataGenerator(
			rescale = 1./255,
			rotation_range=40,
			width_shift_range=0.1,
			height_shift_range=0.1,
			horizontal_flip = True,
			fill_mode='nearest')

# augmentation configuration for testing
test_datagen = ImageDataGenerator(
			rescale = 1./255)




###### preprocessing image #######

# dimensions of our images
img_width, img_height = 140,140 # These are default image dimensions
data_dir = '/home/bilawal/Summer2017/myNetwork/data/all_years_140x140/'
epochs = 10
batch_size = 8 # increase it depending on how fast the gpu runs


if K.image_data_format() == 'channels_first':
	input_shape = (3, img_width, img_height)
else:
	input_shape = (img_width, img_height, 3)

		
# Loads the data from the directory, will try to make a separate class for it
# Returns (X_train, y_train), (X_test, y_test)s

def _load_data(data_dir=data_dir):
	
	labels = os.listdir(data_dir)
	label_counter = 0
	label_lst = list()
	img_lst = list()

	# time to return the images and the labels

	encoder = LabelEncoder()
	for label in labels: 	
		if (not label.startswith('.')):
			img_dir = data_dir + str(label)+"/"
			images = os.listdir(img_dir)
			for img in images:
				if (not img.startswith('.')):
					img2 = imread((img_dir + img)[:])
					img_lst.append(img2)
					label_lst.append(label)

	transformed_label = encoder.fit_transform(label_lst)

	X_data = np.asarray(img_lst)
	y_data = np.asarray(transformed_label, dtype=np.uint8)
	X_data = X_data.transpose(0,3,1,2)

	return (X_data, y_data)


# create the neural network architecture

model = Sequential()
model.add(Convolution2D(32, (3, 3), input_shape=input_shape))
model.add(Activation('relu'))
model.add(MaxPooling2D(pool_size=(2,2)))

model.add(Convolution2D(32, (3, 3)))
model.add(Activation('relu'))
model.add(MaxPooling2D(pool_size=(2,2)))

model.add(Convolution2D(32, (3, 3)))
model.add(Activation('relu'))
model.add(MaxPooling2D(pool_size=(2,2)))

model.add(Flatten())
model.add(Dense(128))
model.add(Activation('relu'))
model.add(Dropout(0.5))
model.add(Dense(22))
model.add(Activation('softmax')) # changed from sigmoid

(X_data, y_data) = _load_data()


#"""

# Preprocess the data

X_data = X_data.astype('float32')
X_data /= 255

#y_data = np.asarray(y_data, dtype='a16')
print y_data.dtype
y_data = keras.utils.to_categorical(y_data, 22)
print y_data.dtype

#"""

model.compile(loss='categorical_crossentropy', # can change to categorical_crossentropy
				optimizer = 'rmsprop', # can use adagrad instead
				metrics = ['accuracy'])

"""

train_generator = train_datagen.flow_from_directory(
					train_data_dir, # make this
					target_size = (img_width, img_height),
					batch_size= batch_size,
					class_mode = 'binary')

test_generator = test_datagen.flow_from_directory(
					test_data_dir,
					target_size = (img_width, img_height),
					batch_size = batch_size,
					class_mode = 'binary')



model.fit_generator (
		(X_data, y_data),
		steps_per_epoch = X_data.shape[0] // batch_size,
		epochs = epochs,
		validation_data = (X_data, y_data),
		validaton_steps = X_data.shape[0] // batch_size)


"""

model.fit(x=X_data, y=y_data, batch_size=32, epochs = 100, validation_split = 0.1, verbose=2)

score = model.evaluate(X_data, y_data, verbose = 0)

# model.save_weights('Saved_Weights.h5')		




