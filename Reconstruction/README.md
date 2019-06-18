###Work on MNIST

All the models mentioned are stored in models folder. All the main_xxx or ipynb files are related results using the models.

To train neural networks, please run main_mnist. The paramters means

- model : the model using in the training process
- max_epoch : max training epoch
- hidden : hidden dimension of the VAE z vector
- batch_size : batch size
- loss_func : selected loss function including MSE & BCE
- learning_rate : given learning rate

A sample code is as follow

```python
python3 main_mnist.py --model=ConvVAE --max_epoch=100 --hidden=32 --loss_func=CELoss
```

###### Need to mention that since I don't have GPUs, thus some of the codes are not supported for running on GPU. If I remember correctly, all the codes on main_mnist doesn't support running on GPUs.

After training, the trained neural networks are stored in checkpoints folder. After that you can run main_variance.py to see the results. Or you can simply see each ipynb.

Imputation-Keras.ipynb implements the imputation task.

Denoise-VAE-keras.ipynb implements the denoising task.

Comparsion_mnist.ipynb implements some comparation mentioned in the report.

Need to mentioned, to see the Conditional VAE result, please train a CVAE model and picutres will automatically store in a figs folder. 

To run GAN please run main_gan.py

### Work on Stanford Dog

To train neural networks, sample code as follow, but please choose the dataset as dog.

```python
python3 main_variance.py --model=LinearVAE --dataset=dog --max_epoch=100 --loss_func=CELoss --hidden=128
```

Also the model will be stored in checkpoints folder. To see the result just run it again with the same model and dataset parameters and the result will be stored in images folder.

To see the result of VAE-GAN, please run code inside VAE-GAN.ipynb or simply run VAE_GAN.py