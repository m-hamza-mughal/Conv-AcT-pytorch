#!/usr/bin/env python
from torchvision.datasets import UCF101
from torchvision import transforms
from torchvision.transforms._transforms_video import CenterCropVideo, NormalizeVideo
import torchvision
import random
import torch
import logging
import sys


logging.basicConfig(stream=sys.stdout, level=logging.DEBUG, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
# torchvision.set_video_backend('video_reader')
torch.manual_seed(0)
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False
random.seed(0)

# mean = [0.485, 0.456, 0.406] 
# std = [0.229, 0.224, 0.225]

def custom_collate(batch):
    # skip audio data
    filtered_batch = []
    for video, _, label in batch:
        filtered_batch.append((video, label))
    return torch.utils.data.dataloader.default_collate(filtered_batch)

def load_test_dataset(video_path, label_dir, num_samples=-1, num_workers=8, video_dim=128, chunk_length=60, num_frames=20, batch_size=32):
    test_tfs = transforms.Compose([
        transforms.Lambda(lambda x: x.permute(0, 3, 1, 2)),
        transforms.Lambda(lambda x: x[::chunk_length//num_frames]), # skip frame
        transforms.Resize((video_dim, video_dim)),
        transforms.Lambda(lambda x: x.float()),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
        
    ])
    if 'UCF-101' in video_path:
        test_tfs = transforms.Compose([
            transforms.Lambda(lambda x: x.permute(3, 0, 1, 2)), # THWC -> CTHW
            transforms.Lambda(lambda x: x/255.0),
            NormalizeVideo([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
            CenterCropVideo(crop_size=(video_dim, video_dim)),
            transforms.Lambda(lambda x: x.permute(1, 0, 2, 3)), # CTHW -> TCHW
            transforms.Lambda(lambda x: x.float()),
        ])
        test_dataset = UCF101(root = video_path, annotation_path = label_dir, transform=test_tfs , train=False, frames_per_clip=num_frames, step_between_clips=5, frame_rate = 15)
    else:
        test_dataset = UCF101(root = video_path, annotation_path = label_dir, transform=test_tfs, train=False, frames_per_clip=chunk_length) #  ,_video_width=video_dim, _video_height=video_dim, 
        
    if num_samples != -1:
        # assert num_samples >= 2000
        test_indices = random.sample(list(range(len(test_dataset))), int(num_samples))
    else:
        test_indices = list(range(len(test_dataset)))
        random.shuffle(test_indices)

    
    test_subset = torch.utils.data.Subset(test_dataset, test_indices)

    test_loader = torch.utils.data.DataLoader(test_subset, batch_size=batch_size, shuffle=True,
                                            num_workers=num_workers,
                                            collate_fn=custom_collate)

    logging.info(f"Total number of test samples: {len(test_subset)}")
    logging.info(f"Total number of (test) batches: {len(test_loader)}")
    
    return test_loader


def load_dataset(video_path, label_dir, num_samples=-1, num_workers=8, video_dim=128, chunk_length=60, num_frames=20, batch_size=32):
    
    train_tfs = transforms.Compose([
        transforms.Lambda(lambda x: x.permute(0, 3, 1, 2)),
        transforms.Lambda(lambda x: x[::chunk_length//num_frames]), # skip second frame
        transforms.Resize((video_dim, video_dim)),
        transforms.RandomResizedCrop(video_dim, scale=(0.8, 1.0)),
        # transforms.CenterCrop(60),
        # transforms.RandomRotation(degrees=10, interpolation=transforms.InterpolationMode.BILINEAR),
        # transforms.GrayScale(),
        # transforms.GaussianBlur(kernel_size=3),
        # transforms.ColorJitter(brightness=.01, contrast=.01, saturation=.01, hue=.01),
        # transforms.RandomPerspective(distortion_scale=0.1),
        # transforms.AugMix(),
        # transforms.RandAugment(),
        # transforms.Lambda(lambda x: x.permute(3, 0, 1, 2)), # THWC -> CTHW
        # transforms.Lambda(lambda x: x/255.0),
        # NormalizeVideo(mean, std),
        # ShortSideScale(size=side_size),
        # CenterCropVideo(crop_size=(crop_size, crop_size)),
        # transforms.Lambda(lambda x: x.permute(1, 0, 2, 3)), # CTHW -> TCHW
        transforms.RandomHorizontalFlip(p=0.5),
        transforms.AugMix(),
        # transforms.RandAugment(),
        transforms.Lambda(lambda x: x.float()),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
        # transforms.Lambda(lambda x: x / 255.),
        
    ])
    val_tfs = transforms.Compose([
        transforms.Lambda(lambda x: x.permute(0, 3, 1, 2)),
        transforms.Lambda(lambda x: x[::chunk_length//num_frames]), # skip second frame
        transforms.Resize((video_dim, video_dim)),
        transforms.Lambda(lambda x: x.float()),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
        # transforms.Lambda(lambda x: x / 255.),s
        
    ])
    train_dataset = UCF101(root = video_path, annotation_path = label_dir, transform=train_tfs, train=True, frames_per_clip=chunk_length)  # ,_video_width=video_dim, _video_height=video_dim, 
    valtest_dataset = UCF101(root = video_path, annotation_path = label_dir, transform=val_tfs, train=False, frames_per_clip=chunk_length) #  ,_video_width=video_dim, _video_height=video_dim, 

    if num_samples != -1:
        # assert num_samples >= 2000
        train_indices = random.sample(list(range(len(train_dataset))), num_samples)
        valtest_indices = random.sample(list(range(len(valtest_dataset))), int(num_samples*0.2))
    else:
        train_indices = list(range(len(train_dataset)))
        valtest_indices = list(range(len(valtest_dataset)))
        random.shuffle(valtest_indices)

    # Warp into Subsets
    train_subset = torch.utils.data.Subset(train_dataset, train_indices)
    test_subset = torch.utils.data.Subset(valtest_dataset, valtest_indices[len(valtest_indices)//2:])
    val_subset = torch.utils.data.Subset(valtest_dataset, valtest_indices[:len(valtest_indices)//2])

   
    train_loader = torch.utils.data.DataLoader(train_subset, batch_size=batch_size, shuffle=True,
                                            num_workers=num_workers, pin_memory=True,
                                            collate_fn=custom_collate)
    val_loader = torch.utils.data.DataLoader(val_subset, batch_size=batch_size, shuffle=True,
                                            num_workers=num_workers, pin_memory=True,
                                            collate_fn=custom_collate)
    test_loader = torch.utils.data.DataLoader(test_subset, batch_size=batch_size, shuffle=True,
                                            num_workers=num_workers,
                                            collate_fn=custom_collate)

    logging.info(f"Total number of train samples: {len(train_subset)}")
    logging.info(f"Total number of test samples: {len(test_subset)}")
    logging.info(f"Total number of val samples: {len(val_subset)}")
    logging.info(f"Total number of (train) batches: {len(train_loader)}")
    logging.info(f"Total number of (test) batches: {len(test_loader)}")
    logging.info(f"Total number of (val) batches: {len(val_loader)}")

    return train_loader, val_loader, test_loader

if __name__=="__main__":
    train_loader, val_loader, test_loader = load_dataset(
        video_path="../datasets/UCF10/",
        label_dir="../datasets/ucf10TrainTestlist/",
        video_dim=128, 
        chunk_length=60, 
        num_frames=20,
        batch_size=32,
        num_samples=-1,
        num_workers=2
    )
    train_iter = iter(train_loader)
    x, y = next(train_iter)
    print("size of train batch: ", x.size(), y.size())
    val_iter = iter(val_loader)
    x, y = next(val_iter)
    print("size of val batch: ", x.size(), y.size())
    test_iter = iter(test_loader)
    x, y = next(test_iter)
    print("size of test batch: ", x.size(), y.size())