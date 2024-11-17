import torch
import time
import json
import evaluate
import argparse
import numpy as np
import pandas as pd
from datasets import Dataset
from transformers import AutoModelForImageClassification, Trainer, TrainingArguments
# from torchvision import models


class dataset():
    def __init__(self, data_path='./data/tensors.pt', label_path='./data/anime_label_map.json', noise_path=None, train_test_split=0):
        self.data_path = data_path
        self.label_path = label_path
        self.noise_path = noise_path
        self.train_test_split=train_test_split

    def get_dataset(self):
        '''
        Loads a pytorch tensor file at a path and returns a huggingface Dataset for
        ease of use in fine-tuning a model.
        '''
        
        # load all data, optionally with noise
        tensors = torch.load(self.data_path)
        if self.noise_path is not None:
            print('Warning: adding noise creates a very, very large dataset. Waiting 5 seconds in case you want to cancel')
            time.sleep(5)
            noise = torch.load(self.noise_path)
            tensors += noise

        all_data = Dataset.from_dict({'pixel_values': [t[0] for t in tensors], 'labels': [t[1] for t in tensors]})
        
        # optionally apply a train/test split
        if self.train_test_split > 0:
            split = all_data.train_test_split(test_size=self.train_test_split, seed=0)
            return split['train'], split['test']
        else:
            return all_data

    def get_labels(self):
        '''
        Return a dictionary mapping labels to encodings
        '''
        with open(self.label_path, 'r') as f:
            labels = json.load(f)

        if self.noise_path is None:
            del labels['11']

        return labels


class classifier():
    '''
    Class defining classifier model, providing methods for training model and making predictions.
    '''
    def __init__(self, model, labels, save_path, learning_rate=5e-5, batch_size=10, num_train_epochs=3, weight_decay=0.1):
        self.learning_rate = learning_rate
        self.per_device_train_batch_size = batch_size
        self.num_train_epochs = num_train_epochs
        self.weight_decay = weight_decay
        self.labels = labels
        self.classifier = AutoModelForImageClassification.from_pretrained(model)
        self.save_path = save_path

    def compute_metrics(self, pred):
        '''
        Helper function for evaluating models using the Trainer module.
        '''
        metric = evaluate.load('accuracy')
        logits, labels = pred
        predictions = np.argmax(logits, axis=-1)

        return metric.compute(predictions=predictions, references=labels)

    def train_model(self, train, test):
        '''
        Perform transfer learning on a pretrained model and save the model
        image to local machine.
        '''

        # update the output layer of the pre-trained model with a linear layer sizing
        # the default ouput down to our n-label classification problem
        self.classifier.classifier = torch.nn.Linear(self.classifier.classifier.in_features, len(self.labels))

        # set device for model execution
        cuda = torch.cuda.is_available()
        mps = torch.backends.mps.is_available()
        device = 'cuda' if cuda else 'mps' if mps else 'cpu'
        self.classifier.to(device)

        # define initial hyperparameters
        args_d = {
            'output_dir': self.save_path + '/model_image',
            'save_strategy': 'no',
            'learning_rate': self.learning_rate,
            'per_device_train_batch_size': self.per_device_train_batch_size,
            'num_train_epochs': self.num_train_epochs,
            'weight_decay': self.weight_decay,
            'eval_strategy': 'steps',
            'eval_steps': 100,
            'logging_steps': 100,
            'per_device_eval_batch_size': self.per_device_train_batch_size,
            'seed': 0
        }
        m_args = TrainingArguments(**args_d)

        # define key training params
        trainer_d = {
            'model': self.classifier,
            'train_dataset': train,
            'eval_dataset': test,
            'compute_metrics': self.compute_metrics,
            'args': m_args         
        }
        trainer = Trainer(**trainer_d)

        # train model and save image
        trainer.train()
        trainer.save_model()

        return trainer
    
    def predict(self, data):
        '''
        Load a saved model and use it to make predictions on a dataset.
        '''

        # import pdb; pdb.set_trace()

        p_args = TrainingArguments(
            # not actually used for predictions, but a required argument
            output_dir=self.save_path + 'predictions'
        )
        p_trainer = Trainer(
            model=AutoModelForImageClassification.from_pretrained(self.save_path + 'model_image'),
            args=p_args
        )
        pred = p_trainer.predict(data)

        # retrieve predictions in form of class probabilities and most likely class
        pred_proba = torch.nn.functional.softmax(torch.tensor(pred.predictions), dim=-1)
        pred_class = [labels[str(l)] for l in (torch.argmax(pred_proba, dim=-1).tolist())]

        # format as pandas dataframes and return complete predictions
        proba_df = pd.DataFrame(columns=list(labels.values()), data=pred_proba.tolist())
        class_df = pd.DataFrame(columns=['predicted_label'], data=pred_class)
        pred_df = pd.concat([class_df, proba_df], axis=1)

        return pred_df


def argparser():
    '''
    Parse whether the script is being ran in training mode, or for daily predictions.
    '''

    # add args for training and predicting
    parser = argparse.ArgumentParser()
    parser.add_argument('--model', type=str, default='google/efficientnet-b3')
    parser.add_argument('--batch_size', type=int, default=32)
    parser.add_argument('--learning_rate', type=float, default=1e-2)
    parser.add_argument('--weight_decay', type=float, default=0.1)
    parser.add_argument('--epochs', type=float, default=2.0)
    parser.add_argument('--save_path', type=str, default='/Users/jonah.krop/Documents/USC/usc_dsci_565_project/')
    parser.add_argument('--train_test_split', type=float, default=0.2)
    parser.add_argument('--data_path', type=str, default='/Users/jonah.krop/Documents/USC/usc_dsci_565_project/data/tensors.pt')
    parser.add_argument('--label_path', type=str, default='/Users/jonah.krop/Documents/USC/usc_dsci_565_project/data/anime_label_map.json')
    parser.add_argument('--noise_path', type=str, default=None)
    
    return parser.parse_args()


if __name__ == '__main__':

    t = int(time.time())

    # parse execution args
    args = argparser()

    valid_models = ['google/efficientnet-b3', 'google/efficientnet-b7', 'mobilenet_v2_1.0_224']
    if args.model not in valid_models:
        raise ValueError(f'for now, please use a model in {valid_models}')

    # initialize train, test, and labels datasets
    dataloader = dataset(
        data_path=args.data_path,
        label_path=args.label_path,
        train_test_split=args.train_test_split,
        noise_path=args.noise_path
    )
    train, test = dataloader.get_dataset()
    labels = dataloader.get_labels()


    # initialize model for training
    model = classifier(
        model=args.model,
        labels=labels,
        save_path=args.save_path,
        learning_rate=args.learning_rate,
        batch_size=args.batch_size,
        num_train_epochs=args.epochs,
        weight_decay=args.weight_decay
    )

    # train model
    train_results = model.train_model(train, test).state.log_history
    train_params = {
        'model': args.model,
        'learning_rate': args.learning_rate,
        'batch_size': args.batch_size,
        'epochs': args.epochs,
        'weight_decay': args.weight_decay,
        'train_test_split': args.train_test_split,
        'added_noise': args.noise_path is not None
    }
    # save step-by-step logs and model params to json file
    with open(args.save_path+f'training_results/model_json_{t}.json', 'w') as f:
        json.dump(train_results, f)
    with open(args.save_path+f'training_results/model_params_{t}.json', 'w') as f:
        json.dump(train_params, f)

    # make predictions with test dataset and save
    pred = model.predict(test) # auto loads model image at save_path
    pd.to_csv(args.save_path + f'predictions_{t}.csv', index=False)