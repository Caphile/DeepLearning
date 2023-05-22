import matplotlib.pyplot as plt
import tensorflow as tf
import pathlib
import time

#---------------------------------------------------------------------------------

enable_argument = True
batch_size = 32
dropout = 0.4
epochs = 50
patience = 10

#---------------------------------------------------------------------------------

print("select dataset 1(flowers) or 2(cirtus) : ", end = "")
opt = int(input())
print("")

if opt == 1:
    dataset_url = "https://storage.googleapis.com/download.tensorflow.org/example_images/flower_photos.tgz"
    fname = 'flowers'
    img_height = 180
    img_width = 180
    num_class = 5

elif opt == 2:
    dataset_url = "https://prod-dcd-datasets-cache-zipfiles.s3.eu-west-1.amazonaws.com/3f83gxmv57-2.zip"
    fname = 'cirtus'
    img_height = 256
    img_width = 256
    num_class = 5

data_dir = tf.keras.utils.get_file(origin = dataset_url, fname = fname, untar = True)
data_dir = pathlib.Path(data_dir)

train_ds = tf.keras.utils.image_dataset_from_directory(
    data_dir,
    validation_split = 0.2,
    subset = "training",
    seed = 123,
    image_size = (img_height, img_width),
    batch_size = batch_size)

val_ds = tf.keras.utils.image_dataset_from_directory(
    data_dir,
    validation_split = 0.2,
    subset = "validation",
    seed = 123,
    image_size = (img_height, img_width),
    batch_size = batch_size)

test_ds = tf.keras.utils.image_dataset_from_directory(
    data_dir,
    validation_split = 0.2,
    subset = "validation",
    seed = 123,
    image_size = (img_height, img_width),
    batch_size = batch_size)

inputs = tf.keras.Input(shape = (img_height, img_width, 3))

data_augmentation = tf.keras.Sequential([tf.keras.layers.RandomFlip("horizontal"), tf.keras.layers.RandomRotation(0.1), tf.keras.layers.RandomZoom(0.2)])
if enable_argument:
    x = data_augmentation(inputs)
else:
    x = inputs
x = tf.keras.layers.Rescaling(1./255)(x)
x = tf.keras.layers.Conv2D(filters = 32, kernel_size = 3, activation = "relu")(x)
x = tf.keras.layers.MaxPooling2D(pool_size = 2)(x)
x = tf.keras.layers.Conv2D(filters = 32, kernel_size = 3, activation = "relu")(x)
x = tf.keras.layers.MaxPooling2D(pool_size = 2)(x)
x = tf.keras.layers.Conv2D(filters = 32, kernel_size = 3, activation = "relu")(x)
x = tf.keras.layers.MaxPooling2D(pool_size = 2)(x)
x = tf.keras.layers.Flatten()(x)
x = tf.keras.layers.Dropout(dropout)(x)
outputs = tf.keras.layers.Dense(num_class, activation = "softmax")(x)

model = tf.keras.Model(inputs = inputs, outputs = outputs)
model.compile(loss = tf.keras.losses.SparseCategoricalCrossentropy(from_logits = False), optimizer = "adam", metrics = ["accuracy"])

callbacks = [
    tf.keras.callbacks.ModelCheckpoint(
        filepath = f"model_{fname}.h5",
        save_best_only = True,
        monitor = "val_loss"),
    tf.keras.callbacks.EarlyStopping(
        monitor = "val_loss",
        patience = patience)
]
print("-------------------------------------------------------")
print("Train\n")
history = model.fit(train_ds, epochs = epochs, validation_data = val_ds, callbacks = callbacks)

accuracy = history.history["accuracy"]
val_accuracy = history.history["val_accuracy"]
loss = history.history["loss"]
val_loss = history.history["val_loss"]
epoch = range(1, len(accuracy) + 1)

print("-------------------------------------------------------")
print("Test\n")
test_model = tf.keras.models.load_model(f"model_{fname}.h5")

print("Evaluate")
test_loss, test_acc = test_model.evaluate(test_ds)

print("Predict")
start_time = time.time()
predictions = test_model.predict(test_ds)
end_time = time.time()
prediction_time = end_time - start_time

print("-------------------------------------------------------")
print("Test result\n")
print(f"Test loss: {test_loss:.3f}")
print(f"Test accuracy: {test_acc:.3f}")
print(f"Test Prediction time: {prediction_time:.3f} seconds")
print("")

plt.figure(f'{fname}_{enable_argument}_{batch_size}_{dropout}_{epochs}_{patience}_acc')
plt.plot(epoch, accuracy, "bo", label = "Training accuracy")
plt.plot(epoch, val_accuracy, "b", label = "Validation accuracy")
plt.title("Training and validation accuracy")
plt.legend()
plt.figure(f'{fname}_{enable_argument}_{batch_size}_{dropout}_{epochs}_{patience}_los')
plt.plot(epoch, loss, "bo", label = "Training loss")
plt.plot(epoch, val_loss, "b", label = "Validation loss")
plt.title("Training and validation loss")
plt.legend()
plt.show()

# The TensorFlow Authors (open source)
# https://colab.research.google.com/github/tensorflow/docs-l10n/blob/master/site/ko/tutorials/load_data/images.ipynb?hl=ko#scrollTo=ufPx7EiCiqgR