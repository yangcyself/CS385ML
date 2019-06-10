## Importing required Libraries
import os
import tensorflow as tf
from tensorflow.contrib.tensorboard.plugins import projector
import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
## Get working directory
PATH = os.getcwd()

## Path to save the embedding and checkpoints generated
LOG_DIR = PATH + '/project-tensorboard/log-1/'
## Load data
df = pd.read_csv("scaled_data.csv",index_col =0)
## Load the metadata file. Metadata consists your labels. This is optional. Metadata helps us visualize(color) different clusters that form t-SNE
metadata = os.path.join(LOG_DIR, 'df_labels.tsv')
# Generating PCA and 
pca = PCA(n_components=50,
         random_state = 123,
         svd_solver = 'auto'
         )
df_pca = pd.DataFrame(pca.fit_transform(df))
df_pca = df_pca.values
## TensorFlow Variable from data
tf_data = tf.Variable(df_pca)
## Running TensorFlow Session
with tf.Session() as sess:
    saver = tf.train.Saver([tf_data])
    sess.run(tf_data.initializer)
    saver.save(sess, os.path.join(LOG_DIR, 'tf_data.ckpt'))
    config = projector.ProjectorConfig()
# One can add multiple embeddings.
    embedding = config.embeddings.add()
    embedding.tensor_name = tf_data.name
    # Link this tensor to its metadata(Labels) file
    embedding.metadata_path = metadata
    # Saves a config file that TensorBoard will read during startup.
    projector.visualize_embeddings(tf.summary.FileWriter(LOG_DIR), config)