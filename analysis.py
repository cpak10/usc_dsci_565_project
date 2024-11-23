import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
import json
import os


def compute_accuracy(time, top_n=2):
    '''
    Compute top 1 and top n accuracy, with grand totals.
    '''

    # load data
    data = pd.read_csv(path+ f'/predictions_{time}.csv')
    proba_cols = data.columns[2:]

    # add custom match_n for top_n accuracy calc
    match = []
    for i in range(len(data)):
        top = data[proba_cols].iloc[i].nlargest(top_n).index.tolist()
        match.append(data['actual'].iloc[i] in top)
    data['match_n'] = match
    data['match'] = data['actual'] == data['predicted_label']

    # prep for grand total rollup
    data['grand_total'] = 'grand total'

    # compute metrics by totals
    g_all = data.groupby(by='grand_total').agg({'match_n': 'mean', 'match': ['mean', 'count']}).reset_index()
    g_all = g_all.rename(columns={'grand_total': 'actual'})

    # compute metrics by label
    g = data.groupby(by='actual').agg({'match_n': 'mean', 'match': ['mean', 'count']}).reset_index()

    # combine and rename columns
    g = pd.concat([g, g_all], axis=0).reset_index(drop=True)
    g.columns = ['anime', f'top-{top_n} accuracy %', 'top-1 accuracy %', 'num_images']
    
    # print(g)
    g.to_csv(path+f'/accuracy_{time}.csv', index=False)


def graph_performance(time):
    '''
    Load model log and plot epoch-by-epoch performance.
    '''
    
    with open(path + f'/model_json_{time}.json', 'r') as f:
        model_json = json.load(f)

    data = []
    for log in model_json:
        if 'loss' in log:
            data.append([log['epoch'], 'train_loss', log['loss']])
        elif 'eval_loss' in log and 'eval_accuracy' in log:
            data.append([log['epoch'], 'eval_loss', log['eval_loss']])
            data.append([log['epoch'], 'eval_accuracy', log['eval_accuracy']])
        elif 'eval_loss' in log:
            data.append([log['epoch'], 'eval_loss', log['eval_loss']])
        else:
            pass
    
    eval_log = pd.DataFrame(columns=['epoch', 'eval', 'loss'], data=data)

    plt.figure()
    sns.lineplot(data=eval_log, x='epoch', y='loss', hue='eval')
    plt.xlabel('Training Epoch')
    plt.ylabel('Model Performance')
    plt.ylim([0, 2.5])
    plt.grid()
    plt.savefig(path + f'/epoch_performance_{time}.png', dpi=200)
    plt.close()


path = '/Users/jonah.krop/Documents/USC/usc_dsci_565_project/training_results'
times = sorted([f.split('.')[0][-10:] for f in os.listdir(path) if os.path.isfile(os.path.join(path, f)) and f[0:11] == 'predictions'])

for time in times:
    graph_performance(time)
    compute_accuracy(time)