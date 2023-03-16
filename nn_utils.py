import numpy as np
import matplotlib.pyplot as plt
import keras_tuner as kt
import os 
import glob
import datetime
from keras.preprocessing.image import ImageDataGenerator
from keras.utils import load_img
from keras.utils import img_to_array
from sklearn.preprocessing import LabelBinarizer
from sklearn.model_selection import train_test_split
from tensorflow import keras
from keras.models import Sequential
from keras.layers import Conv2D, MaxPooling2D, Flatten, Dense
from keras.metrics import categorical_accuracy
from keras.callbacks import EarlyStopping, TensorBoard
from tensorboard.plugins.hparams import api as hp
from tensorflow.keras.applications import ResNet50

def CNNbase_model(input_shape, n_class, learning_rate=0.001):
    model = Sequential(name="Fossil-CNN")
    model.add(Conv2D(32, (3, 3), 
            activation="relu", 
            kernel_initializer="he_uniform", 
            padding="same", 
            input_shape=input_shape,
            name="Input",))
    model.add(MaxPooling2D((2, 2)))
    model.add(Flatten())
    model.add(
        Dense(128, activation='relu', kernel_initializer='he_uniform'))
    model.add(
        Dense(n_class, activation='softmax', name="Output"))

    mod_summary = model.summary()

    # compile model
    opt = keras.optimizers.Adam(lr=learning_rate)
    model.compile(optimizer=opt, 
                  loss='categorical_crossentropy', 
                  metrics=[categorical_accuracy])

    return model, mod_summary

def CNNtrained_model(input_shape, n_class, learning_rate=0.001):
    pre_model = Sequential()
    pre_model.add(
        ResNet50(input_shape=input_shape, include_top=False, 
            weights='imagenet', pooling='max')
            )
    pre_model.add(Dense(n_class, activation='softmax'))
    opt = keras.optimizers.SGD(lr=learning_rate)
    pre_model.compile(optimizer=opt, loss='categorical_crossentropy',
                metrics=[keras.metrics.categorical_accuracy])

    return pre_model

def hyper_tuning(input_shape, n_class):
        class MyHyperModel(kt.HyperModel):
            def build(self, hp):
                filter_1 = hp.Int('conv_filter', min_value=16, max_value=64, step=16)
                kernel_size_1 = hp.Choice('conv_kernel', values=[2, 3, 5])

                model = Sequential()
                model.add(Conv2D(filter_1, kernel_size_1, activation='relu', kernel_initializer='he_uniform', padding='same',
                                 input_shape=input_shape))
                model.add(MaxPooling2D((2, 2)))
                model.add(Flatten())
                for i in range(hp.Int("num_layers", 2, 10)):
                    model.add(Dense(
                            units=hp.Int("units_" + str(i), min_value=32, max_value=128, step=32),
                            activation="relu", kernel_initializer='he_uniform'
                            ))
                model.add(Dense(n_class, activation='softmax'))
                # compile model
                opt = keras.optimizers.Adam(hp.Choice("learning_rate", values=[1e-2, 1e-3, 1e-4]))
                model.compile(optimizer=opt, loss='categorical_crossentropy',
                              metrics=[categorical_accuracy])
                return model

        tuner = kt.Hyperband(MyHyperModel(),
                             objective='val_loss',
                             max_epochs=5,
                             project_name="Fossil_AI")
        return tuner

def add_callbacks():
        log_dir = "logs/fit/" + datetime.datetime.now().strftime("%Y%m%d-%H%M%S")

        my_callbacks = [
            EarlyStopping(monitor='val_categorical_accuracy', mode='max', verbose=1, patience=20),
            #EarlyStopping(monitor='val_loss', mode='min', verbose=1, patience=10),
            #ModelCheckpoint(filepath='model.{epoch:02d}-{val_loss:.2f}.h5'),
            #ClearTrainingOutput(),
            TensorBoard(
                log_dir=log_dir,
                histogram_freq=1,
                update_freq='batch'),
            #hp.KerasCallback(hparams),  # log hparams
        ]
        return my_callbacks

def encode_labels(classlist):
    encoder= LabelBinarizer()
    en_classlist = encoder.fit_transform(classlist)
    return en_classlist

def read_images(images_dir: str) -> tuple:

    Image_List = []
    Labels_list = []
    min_dims = []
    for label in os.listdir(images_dir):
        for image_path in glob.glob(f"{images_dir}/{label}/*.jpg"):
            img = load_img(image_path, color_mode="rgb")
            Image_List.append(img)
           
            img_array = img_to_array(img) 
            min_dims.append(img_array.shape[:2]) 
            Labels_list.append(label)

    # -- resize images
    min_dims = min(min_dims)
    ImageArray_list = []
    for img in Image_List:
        img.thumbnail(min_dims)
        ImageArray_list.append(img_to_array(img))

    # -- 2. split dataset
    ImageArray = np.array(ImageArray_list)
    Labels = encode_labels(Labels_list)
    return ImageArray, Labels

def summarize_diagnostics(history, filename="./summary.jpg"):

        _, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 8), sharex=True)

        # -- plot loss
        ax1.set_title('Categorical_Crossentropy Loss')
        ax1.semilogy(history.history['loss'], color='blue', label='train')
        ax1.semilogy(history.history['val_loss'], color='orange', label='test')
        ax1.set_ylabel('loss', fontsize=10)
        ax1.legend()

        # -- plot accuracy
        ax2.set_title('Categorical Accuracy')
        ax2.plot(history.history['categorical_accuracy'], color='blue', label='train')
        ax2.plot(history.history['val_categorical_accuracy'], color='orange', label='test')
        ax2.set_ylabel('c. accuracy', fontsize=10)
        ax2.set_xlabel("epoch", fontsize=10)
        ax2.legend()

        # -- save plot to file
        plt.savefig(filename)
        #plt.show()
        #plt.close()
        return plt.show()

def sample_dataset(X, y, size=10, random_state=None):
    np.random.seed(random_state)
    indices = np.random.choice(a=np.arange(len(y)), size=size, replace=False)
    return X[indices], y[indices]

def plot_prediction(x, y_true, y_pred, filename="predictions.jpg"):

    labels = np.arange(y_true.shape[1])
    fig, axis = plt.subplots(3, 3, figsize=(8, 8), sharex=True, sharey=True)
    axis = axis.flatten()
    for i in range(9):
        label_true = labels[y_true[i] == 1][0]
        label_pred = labels[np.argmax(y_pred[i])]
        axis[i].imshow(x[i] / 255, cmap="binary", aspect="equal")
        axis[i].set_title(f"true class: {label_true} - pred. class: {label_pred}",
            fontsize=10, color="g" if label_true == label_pred else "r")
        axis[i].tick_params(bottom=False, labelbottom=False, left=False, labelleft=False)

        # bbox = axis[i].get_position()
        # ax = fig.add_axes(bbox)
        # ax.patch.set_alpha(0)
        # ax.bar(x=labels, height=y_pred[i], 
        #     width=1, color="gold", edgecolor="None", alpha=0.5)
        # ax.set_xlim(-0.5, labels[-1]+0.5)
        # ax.set_ylim(0., 1.05)
        # ax.tick_params(left=False, labelleft=False)
        # if i in [0, 3, 5]:
        #     ax.set_ylabel("prediction PDF")
        # if i > 5:
        #     ax.set_xlabel("class #")
        #     ax.set_xticks(labels)
        #     ax.tick_params(axis="x", rotation=90, labelsize=10)
        # else:
        #     ax.set_xticks(labels)
        #     ax.tick_params(labelbottom=False)

    fig.savefig(filename, dpi=300)
    plt.tight_layout()
    #plt.close()
    return plt.show()


def main():

    # -- 1. read data
    input_dir = "data/ImageClasses"

    Image_List = []
    Labels_list = []
    min_dims = []
    for label in os.listdir(input_dir):
        for image_path in glob.glob(f"{input_dir}/{label}/*.jpg"):
            img = load_img(image_path, color_mode="grayscale")
            Image_List.append(img)
           
            img_array = img_to_array(img)
            min_dims.append(img_array.shape[:2]) 
            Labels_list.append(label)

    # -- 2. resize images
    min_dims = min(min_dims)
    ImageArray_list = []
    for img in Image_List:
        img.thumbnail(min_dims)
        ImageArray_list.append(img_to_array(img))

    # -- 3. split dataset
    number_of_labels = len(np.unique(Labels_list))
    Labels = encode_labels(classlist=Labels_list)
    ImageArray = np.array(ImageArray_list)

    X_train, X_test, y_train, y_test = \
        train_test_split(ImageArray, Labels, test_size=0.3, random_state=42)

    datagen = ImageDataGenerator(rescale=1. / 255,
                                validation_split=0.2,
                                rotation_range=40,
                                width_shift_range=0.2,
                                height_shift_range=0.2,
                                shear_range=0.2,
                                zoom_range=0.2,
                                horizontal_flip=True,
                                fill_mode='nearest')
    
    # -- 4. generate CNN base model
    input_shape = ImageArray.shape[1:]
    model, _ = CNNbase_model(input_shape=input_shape, n_class=number_of_labels)
    
    # -- 4.1 training
    history = model.fit(
        datagen.flow(X_train, y_train, batch_size=32, subset="training"), 
        validation_data=datagen.flow(X_train, y_train, subset="validation"),
        steps_per_epoch=None, 
        epochs=50, 
        validation_steps=None,
        callbacks=add_callbacks())

    # -- 4.2 testing
    eval_result = model.evaluate(X_test, y_test)

    val_acc_per_epoch = history.history['val_categorical_accuracy']
    best_epoch = val_acc_per_epoch.index(max(val_acc_per_epoch)) + 1
    print(f"""
            Evaluation of test data [test loss, test accuracy] is {eval_result}.
        """)

    # -- 5. generate CNN model with hyperparameter tuning
    tuner = hyper_tuning(input_shape=input_shape, n_class=number_of_labels)
    tuner.search(
        datagen.flow(X_train, y_train, batch_size=32, subset="training"), 
        validation_data=datagen.flow(X_train, y_train, subset="validation"),
        steps_per_epoch=None, 
        epochs=50, 
        validation_steps=None,
        callbacks=add_callbacks())

    tuner.results_summary()

    # -- 5.1 Get the optimal hyperparameters
    best_hps = tuner.get_best_hyperparameters(1)[0]
    print(f"""
            The hyperparameter search is complete. The optimal number of
            layers {best_hps.get('num_layers')} and the optimal learning rate for the optimizer
            is {best_hps.get('learning_rate')}. Best Conv. Filter: {best_hps.get('conv_filter')}. 
            Best Conv. Kernel: {best_hps.get('conv_kernel')}. 
        """)

    # -- 5.2 get best architecture for a new model
    best_model = tuner.hypermodel.build(best_hps)

    # -- 5.3 training
    history = best_model.fit(
        datagen.flow(X_train, y_train, batch_size=32, subset="training"), 
        validation_data=datagen.flow(X_train, y_train, subset="validation"),
        steps_per_epoch=None, 
        epochs=50, 
        validation_steps=None,
        callbacks=add_callbacks())
    
    # -- 5.4 testing
    eval_result = best_model.evaluate(X_test, y_test)
    
    val_acc_per_epoch = history.history['val_categorical_accuracy']
    best_epoch = val_acc_per_epoch.index(max(val_acc_per_epoch)) + 1
    print(f"""
            Best epoch is {best_epoch}. 
            Evaluation of test data [test loss, test accuracy] is {eval_result}.
        """)

    # -- 6. Retrain the model with optimal number of epoch
    hypermodel = tuner.hypermodel.build(best_hps)
    hypermodel.fit(datagen.flow(X_train, y_train, batch_size=32, subset="training"), 
        validation_data=datagen.flow(X_train, y_train, subset="validation"),
        steps_per_epoch=None, 
        epochs=best_epoch, 
        validation_steps=None,
        callbacks=add_callbacks())
    return

if __name__ == "__main__":
    main()