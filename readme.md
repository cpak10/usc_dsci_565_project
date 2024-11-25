# Animation Classification Through Various CNN Architectures

## Data Initialization
In effort to standardize data for model development, please follow the instructions below.

1. Initialize virtual environment on python 3.11
    1. `python3.11 -m venv .venv`
    2. `. .venv/bin/activate`
2. `pip install -r requirements.txt`
3. `python preprocessing.py`

These steps should initialize a `/data` folder in your current directory and create two .pt files. You should only have to run this code once per environment.

1. `tensors.pt` contains all of the training images in tensor form
2. `tensors_noise.pt` contains everything in `tensors.pt` and also adds in random images as noise

You can also run `preprocessing_check.ipynb` to ensure the data was downloaded correctly.

## Training Models
Models can be trained using a one line command that utilizes the packages from Hugging Face.

`python transfer_learning.py --model [INSET MODEL NAME] --directory [INSERT WORKING DIRECTORY]`

This call allows for various flags to alter the training process.

### model
Hugging Face location of specific model to train.

* Type: `str`
* Default: `google/efficientnet-b3`

### pretrained
No argument required. Add flag if model should be initialized with pretrained weights.

### add_noise
No argument required. Add flag if noise should be added into training process.

### batch_size
* Type: `int`
* Default: `32`

### learning_rate
* Type: `float`
* Default: `1e-2`

### weight_decay
* Type: `float`
* Default: `0.1`

### epochs
* Type: `float`
* Default: `2.0`

### directory
Location of working directory. Can be set to `.` if current directory is working.

* Type: `str`
* Default: `/Users/jonah.krop/Documents/USC/usc_dsci_565_project`

### train_test_split
* Type: `float`
* Default: `0.2`

## GPU Training
If running GPU training on local, download CUDA toolkit [here](https://developer.nvidia.com/cuda-11-7-0-download-archive) and cuDNN v8.9.7 [here](https://developer.nvidia.com/rdp/cudnn-archive).

Additionally, initialize a new virtual environment and install `requirements_gpu.txt`. There is a chance you may need to run the following for `torch` and `torchvision`.

`pip install torch torchvision --index-url https://download.pytorch.org/whl/cu117`

## Model Analysis
Select summary statistics can be generated for each run of a trained model with a one line call.

`python analysis.py --resultdir [INSERT PATH TO RESULTS]`
