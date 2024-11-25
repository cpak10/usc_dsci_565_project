import os
import json
import shutil
import zipfile
from time import time
import torch
from torchvision import transforms
from datasets import load_dataset
from huggingface_hub import hf_hub_download


def download_data_bangumi(title: str, label: int, n: int) -> list:
    '''
    Download data from huggingface BangumiBase (https://huggingface.co/BangumiBase)

    Parameters
    ----------
    title : str
        title of the anime
    label : int
        label for the anime, for supervised learning
    n : int
        number of images to download

    Returns
    -------
    tensors : list[tuple[torch.Tensor]]
        training data with image tensor and labels
    '''
    transform = transforms.Compose([
        transforms.Resize((256, 256)),
        transforms.ToTensor()])

    tensors = []

    hf_hub_download(repo_id=f'BangumiBase/{title}', filename='all.zip', repo_type='dataset',
                    local_dir='./data')
    with zipfile.ZipFile('./data/all.zip', 'r') as zipped:
        zipped.extractall('./data/images')

    dataset = load_dataset('imagefolder', data_dir='./data/images', split='train')
    iterable_dataset = dataset.to_iterable_dataset(num_shards=100)
    iterable_dataset_shuffled = iterable_dataset.shuffle(buffer_size=100)

    for item, data in enumerate(iterable_dataset_shuffled):
        if item == n:
            break
        image = data['image']
        image_tensor = transform(image)
        tensors.append((image_tensor, label))

        if (item + 1) % 100 == 0:
            print(f'{title} - processed {item+1} images')
    print(f'finished downloading {title}')

    os.remove('./data/all.zip')
    shutil.rmtree('./data/images', ignore_errors=True)
    shutil.rmtree('./data/.cache')

    return tensors


def save_tensors(data: list[tuple[torch.Tensor]], target: str) -> None:
    '''
    Creates a .pt file for the tensors. Adds to existing file if needed.

    Parameters
    ----------
    data : list[tuple[torch.Tensor]]
        the image tensor and label
    target : str
        file name
    
    Returns
    -------
    None
    '''
    target_path = f'./data/{target}.pt'
    if os.path.isfile(target_path):
        previous_data = torch.load(target_path, weights_only=True)
        new_data = previous_data + data
        torch.save(new_data, target_path)
    else:
        torch.save(data, target_path)


def create_data_dir(dir_name: str) -> None:
    '''
    Create new data directory, delete if exists

    Parameters
    ----------
    dir_name : str
        name of directory
    
    Returns
    -------
    None
    '''
    if os.path.exists(dir_name):
        shutil.rmtree(dir_name)

    os.makedirs(dir_name)

    gitignore_path = os.path.join(dir_name, '.gitignore')
    with open(gitignore_path, 'wt', encoding='utf-8') as gitignore_file:
        gitignore_file.write('*\n')


if __name__ == '__main__':

    start_time = time()

    animes = ['haikyuu', 'jujutsukaisen', 'chainsawman', 'sousounofrieren', 'spyxfamily',
              'bluelock', 'skiptoloafer', 'kimetsunoyaibayuukakuhen',
              'deaddeaddemonsdededededestruction', 'durarara']

    create_data_dir('./data')
    for i, anime in enumerate(animes):
        all_tensors = download_data_bangumi(anime, i, 1_000)
        save_tensors(all_tensors, 'tensors')
    print('all in-sample downloads finished')

    label_map = dict(enumerate(animes))
    label_map[11] = 'noise'
    with open('./data/anime_label_map.json', 'w') as f:
        json.dump(label_map, f)

    out_of_samples = ['nurarihyonnomago', 'happysugarlife', 'wonderfulprecure',
                      'shinmaiossanboukenshasaikyoupartynishinuhodokitaeraretemutekininaru',
                      'uruseiyatsura2022', 'higurashinonakukoroni', 'konosuba', 'hametsunooukoku',
                      'attackontitan', 'unlimitedfafnir']

    tensors_prev = torch.load('./data/tensors.pt', weights_only=True)
    save_tensors(tensors_prev, 'tensors_noise')

    for noise in out_of_samples:
        all_tensors = download_data_bangumi(noise, 11, 100)
        save_tensors(all_tensors, 'tensors_noise')
    print('all out-sample downloads finished')

    end_time = time()
    time_taken = (end_time - start_time) // 60
    print('total time:', time_taken, 'min')
