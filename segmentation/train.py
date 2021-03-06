from keras.callbacks import TensorBoard
import json
from data_utils.data_loader import image_segmentation_generator, verify_segmentation_dataset
from models import model_from_name
from config import *
from models import fcn
import os
import six
import glob


def find_latest_checkpoint(checkpoints_path):
    paths = glob.glob(checkpoints_path + ".*")
    maxep = -1
    r = None
    for path in paths:
        ep = int(path.split('.')[-1])
        if ep > maxep:
            maxep = ep
            r = path
    return r, maxep


def train(model,
          train_images,
          train_annotations,
          input_height=None,
          input_width=None,
          n_classes=None,
          verify_dataset=True,
          checkpoints_path=None,
          epochs=5,
          batch_size=2,
          validate=False,
          val_images=None,
          val_annotations=None,
          val_batch_size=2,
          auto_resume_checkpoint=False,
          load_weights=None,
          steps_per_epoch=512,
          optimizer_name='adadelta',
          callbacks=None):

    if isinstance(model, six.string_types):
        # check if user gives models name insteead of the models object
        # create the models from the name
        assert (not n_classes is None), "Please provide the n_classes"
        if (not input_height is None) and (not input_width is None):
            model = model_from_name[model](n_classes, input_height=input_height, input_width=input_width)
        else:
            model = model_from_name[model](n_classes)

    n_classes = model.n_classes
    input_height = model.input_height
    input_width = model.input_width
    output_height = model.output_height
    output_width = model.output_width

    model.summary()
    if validate:
        assert not (val_images is None)
        assert not (val_annotations is None)

    if not optimizer_name is None:
        model.compile(loss='categorical_crossentropy',
                      optimizer=optimizer_name,
                      metrics=['accuracy'])

    if not checkpoints_path is None:
        open(checkpoints_path+"_config.json", "w").write(json.dumps({
            "model_class": model.model_name,
            "n_classes": n_classes,
            "input_height": input_height,
            "input_width": input_width,
            "output_height": output_height,
            "output_width": output_width
        }))

    if (not (load_weights is None)) and len(load_weights) > 0:
        print("Loading weights from ", load_weights)
        model.load_weights(load_weights)

    latest_ep = -1
    if auto_resume_checkpoint and (not checkpoints_path is None):
        latest_checkpoint, latest_ep = find_latest_checkpoint(checkpoints_path)
        if not latest_checkpoint is None:
            print("Loading the weights from latest checkpoint ", latest_checkpoint)
            model.load_weights(latest_checkpoint)

    if verify_dataset:
        print("Verifying train dataset")
        verify_segmentation_dataset(train_images, train_annotations, n_classes)
        if validate:
            print("Verifying val dataset")
            verify_segmentation_dataset(val_images, val_annotations, n_classes)

    train_gen = image_segmentation_generator(train_images,
                                             train_annotations,
                                             batch_size,
                                             n_classes,
                                             input_height,
                                             input_width,
                                             output_height,
                                             output_width)

    if validate:
        val_gen = image_segmentation_generator(val_images,
                                               val_annotations,
                                               val_batch_size,
                                               n_classes,
                                               input_height,
                                               input_width,
                                               output_height,
                                               output_width)


    if not validate:
        for ep in range(latest_ep + 1, latest_ep + 1 + epochs):
            print("Starting Epoch ", ep)
            model.fit_generator(train_gen,
                                steps_per_epoch,
                                epochs=1,
                                callbacks=callbacks)
            if not checkpoints_path is None:
                model.save_weights(checkpoints_path + "." + str(ep))
                print("saved ", checkpoints_path + ".models." + str(ep))
            print("Finished Epoch", ep)
    else:
        for ep in range(latest_ep + 1, latest_ep + 1 + epochs):
            print("Starting Epoch ", ep)
            model.fit_generator(train_gen,
                                steps_per_epoch,
                                validation_data=val_gen,
                                validation_steps=200,
                                callbacks=callbacks,
                                epochs=1)

            if not checkpoints_path is None:
                model.save_weights(checkpoints_path + "." + str(ep))
                print("saved ", checkpoints_path + ".models." + str(ep))
            print("Finished Epoch", ep)


if __name__ == '__main__':
    train_images_dir = DATA_PATH + "/train_images"
    val_images_dir = DATA_PATH + "/val_images"
    train_labels_dir = DATA_PATH + "/train_labels"
    val_labels_dir = DATA_PATH + "/val_labels"
    tensorboard = TensorBoard(log_dir='./logs/%s/' % MODEL_NAME)
    train(model=MODEL_NAME,
          train_images=train_images_dir,
          train_annotations=train_labels_dir,
          input_height=INPUT_HEIGHT,
          input_width=INPUT_WIDTH,
          n_classes=NUM_CLASS,
          verify_dataset=False,
          checkpoints_path="checkpoints/" + MODEL_NAME,    # don't add '/' in the end
          epochs=EPOCH,
          batch_size=BATCH_SIZE,
          validate=True,
          val_images=val_images_dir,
          val_annotations=val_labels_dir,
          val_batch_size=BATCH_SIZE,
          auto_resume_checkpoint=AUTO_RESUME,
          load_weights=None,
          steps_per_epoch=STEPS_PER_EPOCH,
          optimizer_name=OPTIMIZER,
          callbacks=[tensorboard]
          )


