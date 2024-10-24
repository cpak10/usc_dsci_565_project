# Animation Classification Through Various CNN Architectures

## Data Initialization
In effort to standardize data for model development, please follow the instructions below.

### Local Setup
1. `pip install -r requirements.txt`
2. `python preprocessing.py`

### Google Colab Setup
1. Upload `preprocessing.py` to working directory.
2. `!pip install datasets`
3. `!python preprocessing.py`

For more information on setting up virtual machines for Google Colab please [see the below section](#machine-initialization).

Both of these steps should initialize a data folder in your current directory and create two .pt files. You should only have to run this code once per environment. Once the .pt files are created, you can then run the following code to load the data for model development.

```
data = torch.load('./data/tensors.pt', weights_only=True)

train_percent = 0.9
test_percent = 0.1
train_dataset, test_dataset = torch.utils.data.random_split(data, [train_percent, test_percent])

train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=32, shuffle=True)
test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=32, shuffle=True)
```

You can also run `preprocessing_check.ipynb` to ensure the data was downloaded correctly.

## Machine Initialization

The default memory allocations for Google Colab is not sufficient for dealing with the entire dataset. You can increase the memory allocations through a custom virtual machine initialized on Google Cloud Platform. Follow the [instructions here](https://research.google.com/colaboratory/marketplace.html). Through the custom machine, you can increase memory, storage, and also add GPUs. Memory settings above 30GB should suffice for working with the data. Training models may require larger memory settings and possibly GPUs.