import matplotlib.pyplot as plt
import numpy as np
import torchvision
import torch
import os


def show_batch(batch):
    imgs = torchvision.utils.make_grid(batch)
    imgs = imgs * 0.3234 + 0.1187
    npimgs = imgs.numpy()
    plt.imshow(np.transpose(npimgs, (1, 2, 0)))
    plt.show()


def load(data_path, _32=False):
    if _32:
        transforms = torchvision.transforms.Compose([
            torchvision.transforms.CenterCrop(64),
            torchvision.transforms.Resize(32),
            torchvision.transforms.ToTensor(),
            torchvision.transforms.Lambda(lambda x: x[:1]),
            torchvision.transforms.Normalize((0.1187,),
                                             (0.3234,))
        ])
    else:
        transforms = torchvision.transforms.Compose([
            torchvision.transforms.CenterCrop(64),
            torchvision.transforms.ToTensor(),
            torchvision.transforms.Lambda(lambda x: x[:1]),
            torchvision.transforms.Normalize((0.1187,),
                                             (0.3234,))
        ])

    data = {x: torchvision.datasets.ImageFolder
            (os.path.join(data_path, x), transform=transforms)
            for x in ['train', 'test']}

    data['train_full'] = data['train']

    valid_len = int(0.2 * len(data['train_full'].samples))
    train_len = len(data['train_full'].samples) - valid_len

    data['train'], data['validation'] =\
        torch.utils.data.dataset.random_split(dataset=data['train_full'],
                                              lengths=[train_len, valid_len])

    val_sampler = torch.utils.data.sampler\
        .SubsetRandomSampler(data['validation'].indices)
    train_sampler = torch.utils.data.sampler\
        .SubsetRandomSampler(data['train'].indices)
    return data, train_sampler, val_sampler


def tell(string, data_path):
    data_dict, _, _ = load(data_path)

    train_sampler = torch.utils.data.sampler \
        .SubsetRandomSampler(data_dict['train'].indices)

    train_loader = torch.utils.data\
        .DataLoader(data_dict['train_full'], batch_size=256,
                    sampler=train_sampler, num_workers=0)

    indices = [data_dict['test'].class_to_idx[letter]
               for letter in string]

    dataiter = iter(train_loader)
    images, labels = dataiter.next()

    word = []
    for idx in indices:
        for image, label in zip(images, labels):
            if label == idx:
                word.append(image)
                break
    word = torch.stack(word)
    show_batch(word)


def rand_img(n, data_path):
    data_dict, _, _ = load(data_path)
    train_sampler = torch.utils.data.sampler\
        .SubsetRandomSampler(data_dict['train'].indices)
    train_loader = torch.utils.data\
        .DataLoader(data_dict['train_full'], batch_size=n,
                    sampler=train_sampler, num_workers=4)
    dataiter = iter(train_loader)
    plt.rcParams['figure.figsize'] = (10, 8)
    images, labels = dataiter.next()
    show_batch(images)
    print([data_dict['test'].classes[i] for i in labels])
