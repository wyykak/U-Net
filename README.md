# U-Net
Split medical images with U-Net

## Dependency
Keras and Numpy

## How to use
1. Put your image dataset in _data_ directory.
2. Run _data.py_.
3. Run _unet.py_.
4. Get the results in _results_ directory.

## How to train for more epochs
Just run _unet.py_ again. It will automatically load previously trained model and continue training it.
If you want to train a brand-new model, just delete _unet.hdf5_, which is the checkpoint file.
