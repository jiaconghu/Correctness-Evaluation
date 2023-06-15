from torchvision import datasets
from torchvision import transforms
from torch.utils.data import DataLoader

imagenet_transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225))
])


def load(data_dir, data_type):
    # data = datasets.ImageNet(root=data_dir,
    #                          split='train',
    #                          transform=imagenet_transform)
    data = datasets.ImageFolder(root=data_dir,
                                transform=imagenet_transform)

    data_loader = DataLoader(dataset=data,
                             batch_size=256,
                             num_workers=4,
                             shuffle=True)

    return data_loader
