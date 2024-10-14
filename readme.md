# Animation Classification Through Various CNN Architectures

## Data Initialization
In effort to standardize the training and testing data for model development, please follow the instructions below.

### Local Setup
1. `pip install -r requirements.txt`
2. `python preprocessing.py`

### Google Colab Setup
1. Upload `preprocessing.py` to working directory.
2. `!pip install datasets`
3. `!python preprocessing.py`

Both of these steps should initialize a data folder in your current directory and create four .pt files. You should only have to run this code once per environment. Once the .pt files are created, you can then run the following code to load the data for model development.

```
training_data = torch.load('./data/train_tensors.pt', weights_only=True)
trainloader = torch.utils.data.DataLoader(training_data, batch_size=32, shuffle=True)
```

You can also run `preprocessing_check.ipynb` to ensure the data was downloaded correctly.
