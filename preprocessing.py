import os
import shutil
import torch
from torchvision import transforms
from datasets import load_dataset


def download_data_bangumi(title: str, label: int, n: int) -> tuple[list]:
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
    train : list[tuple[torch.Tensor]]
        training data with image tensor and labels
    test : list[tuple[torch.Tensor]]
        testing data with image tensor and labels
    '''
    transform = transforms.Compose([
        transforms.Resize((256, 256)),
        transforms.ToTensor()])

    train = []
    test = []

    dataset = load_dataset(f'BangumiBase/{title}', split='train', streaming=True)
    for item, data in enumerate(dataset):
        if item == n:
            break
        image = data['image']
        image_tensor = transform(image)
        if item % 5 != 0:
            train.append((image_tensor, label))
        else:
            test.append((image_tensor, label))

        if (item + 1) % 100 == 0:
            print(f'{title} - downloaded {item+1} images')
    print(f'finished downloading {title}')

    return (train, test)


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

    animes = ['haikyuu', 'jujutsukaisen', 'chainsawman', 'sousounofrieren', 'spyxfamily',
              'bluelock', 'skiptoloafer', 'kimetsunoyaibayuukakuhen',
              'deaddeaddemonsdededededestruction', 'durarara']

    create_data_dir('./data')
    for i, anime in enumerate(animes):
        if i == 2:
            break
        train_tensors, test_tensors = download_data_bangumi(anime, i, 1_000)
        save_tensors(train_tensors, 'train_tensors')
        save_tensors(test_tensors, 'test_tensors')
    print('all downloads finished')
