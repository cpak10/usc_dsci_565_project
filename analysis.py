import argparse
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


def make_heatmap(time):
    '''
    Make a heat map with predicted labels on the x axis, and actual labels on y.
    Show % of total events that fall in each bucket.
    '''
    
    data = pd.read_csv(path+f'/predictions_{time}.csv')

    # unique categories
    cats = sorted(list(data['actual'].unique()))

    # prep dataset 
    permu = pd.MultiIndex.from_product([cats, cats], names=['actual', 'predicted_label'])
    grid = pd.DataFrame(index=permu).reset_index()
    full = pd.merge(grid, data.groupby(by=['actual', 'predicted_label']).size().reset_index(name='num'), how='left')
    full = pd.merge(full, data.groupby(by=['actual']).size().reset_index(name='label_num'), how='left')
    full['num'] = full['num'].fillna(0)
    full['pct'] = full['num']/full['label_num']

    # build heatmap
    heatmap = full.pivot(index='actual', columns='predicted_label', values='pct')
    sns.heatmap(heatmap, cmap='Blues', cbar_kws={'label': 'pct'}, annot_kws={'size': 8}, annot=True, linewidths=0.2, linecolor='black', fmt='.2f', xticklabels=1, yticklabels=1)
    plt.xticks(fontsize=6)
    plt.yticks(fontsize=6)
    plt.xlabel('Predicted Category', fontsize=8)
    plt.ylabel('Actual Category', fontsize=8)
    plt.savefig(path+f'/heatmap_{time}.png', dpi=300, bbox_inches='tight')
    plt.close()


def summarize(time):
    '''
    Summarize a set of predictions in one line, with accuracy by category
    and training parameters for quick reference.
    '''

    # load data
    data = pd.read_csv(path+f'/predictions_{time}.csv')
    data['match'] = data['actual'] == data['predicted_label']

    # load model params
    with open(path+f'/model_params_{time}.json', 'r') as f:
        params = json.load(f)

    # load model log
    with open(path+f'/model_json_{time}.json', 'r') as f:
        log = json.load(f)

    # build row
    summ = [
        time,
        params['model'],
        params['learning_rate'],
        params['batch_size'],
        params['epochs'],
        params['weight_decay'],
        params['train_test_split'],
        params['pretrained'],
        np.mean(data['match']),
        np.mean(data['match'][data['actual'] == 'haikyuu']),
        np.mean(data['match'][data['actual'] == 'jujutsukaisen']),
        np.mean(data['match'][data['actual'] == 'chainsawman']),
        np.mean(data['match'][data['actual'] == 'sousounofrieren']),
        np.mean(data['match'][data['actual'] == 'spyxfamily']),
        np.mean(data['match'][data['actual'] == 'bluelock']),
        np.mean(data['match'][data['actual'] == 'skiptoloafer']),
        np.mean(data['match'][data['actual'] == 'kimetsunoyaibayuukakuhen']),
        np.mean(data['match'][data['actual'] == 'deaddeaddemonsdededededestruction']),
        np.mean(data['match'][data['actual'] == 'durarara']),
        np.mean(data['match'][data['actual'] == 'noise']),
        log[-1]['train_runtime']/60
    ]

    return summ

    


parser = argparse.ArgumentParser()
parser.add_argument('--resultdir', type=str,
    default='/Users/jonah.krop/Documents/USC/usc_dsci_565_project/training_results')
args = parser.parse_args()
path = args.resultdir
times = sorted([f.split('.')[0][-10:] for f in os.listdir(path) if os.path.isfile(os.path.join(path, f)) and f[0:11] == 'predictions'])

# go thru every prediction file and do stuff
combined = []
for time in times:
    graph_performance(time)
    compute_accuracy(time)
    make_heatmap(time)
    combined.append(summarize(time))

# make the combined, summarized dataset
combined_cols = [
    'run_timestamp',
    'model',
    'learning_rate',
    'batch_size',
    'epochs',
    'weight_decay',
    'train_test_split',
    'pretrained',
    'accuracy',
    'haikyuu_accuracy',
    'jujutsukaisen_accuracy',
    'chainsawman_accuracy',
    'sousounofrieren_accuracy',
    'spyxfamily_accuracy',
    'bluelock_accuracy',
    'skiptoloafer_accuracy',
    'kimetsunoyaibayuukakuhen_accuracy',
    'deaddeaddemonsdededededestruction_accuracy',
    'durarara_accuracy',
    'noise_accuracy',
    'runtime_mins'
]
df = pd.DataFrame(data=combined, columns=combined_cols)
df.to_csv(path+'/combined_results.csv', index=False)