from torchvision.datasets import UCF101
from torchvision import transforms
import torchvision
import random
import torch


torchvision.set_video_backend('video_reader')
random.seed(0)


def load_dataset(video_path, label_dir, num_samples=None, video_dim=128, chunk_length=30, num_frames=15, batch_size=16):
    tfs = transforms.Compose([
        transforms.Lambda(lambda x: x.permute(0, 3, 1, 2)),
        transforms.Lambda(lambda x: x[::chunk_length//num_frames]), # skip second frame
        transforms.RandomHorizontalFlip(p=0.5),
        # transforms.RandomResizedCrop(256, scale=(0.8, 1.0)),
        # transforms.CenterCrop(60),
        # transforms.RandomRotation(degrees=10, interpolation=transforms.InterpolationMode.BILINEAR),
        # transforms.GrayScale(),
        # transforms.GaussianBlur(kernel_size=3),
        # transforms.ColorJitter(brightness=.2, hue=.1),
        # transforms.RandomPerspective(distortion_scale=0.1),
        # transforms.AugMix(),
        # transforms.RandAugment(),

        transforms.Lambda(lambda x: x / 255.),
        transforms.Lambda(lambda x: x.float()),
    ])
    train_dataset = UCF101(root = video_path, annotation_path = label_dir, transform=tfs ,_video_width=video_dim, _video_height=video_dim, train=True, frames_per_clip=chunk_length)
    # valtest_dataset = UCF101(root = video_path, annotation_path = label_dir, transform=transforms ,_video_width=video_dim, _video_height=video_dim, train=False, frames_per_clip=frame_length)

    if num_samples is not None:
        assert num_samples >= 3000
        train_indices = random.sample(list(range(len(train_dataset))), num_samples)
        # val_indices = random.sample(list(range(len(valtest_dataset))), len(train_dataset)//30)
    else:
        train_indices = list(range(len(train_dataset)))

    # Warp into Subsets
    train_subset = torch.utils.data.Subset(train_dataset, train_indices[:-2000])
    test_subset = torch.utils.data.Subset(train_dataset, train_indices[-1000:])
    val_subset = torch.utils.data.Subset(train_dataset, train_indices[-2000:-1000])

    def custom_collate(batch):
        # skip audio data
        filtered_batch = []
        for video, _, label in batch:
            filtered_batch.append((video, label))
        return torch.utils.data.dataloader.default_collate(filtered_batch)

   
    train_loader = torch.utils.data.DataLoader(train_subset, batch_size=batch_size, shuffle=True,
                                            num_workers=2, pin_memory=True,
                                            collate_fn=custom_collate)
    val_loader = torch.utils.data.DataLoader(val_subset, batch_size=batch_size, shuffle=True,
                                            num_workers=2, pin_memory=True,
                                            collate_fn=custom_collate)
    test_loader = torch.utils.data.DataLoader(test_subset, batch_size=batch_size, shuffle=True,
                                            num_workers=2,
                                            collate_fn=custom_collate)

    print(f"Total number of train samples: {len(train_subset)}")
    print(f"Total number of test samples: {len(test_subset)}")
    print(f"Total number of val samples: {len(val_subset)}")
    print(f"Total number of (train) batches: {len(train_loader)}")
    print(f"Total number of (test) batches: {len(test_loader)}")
    print(f"Total number of (val) batches: {len(val_loader)}")

    return train_loader, val_loader, test_loader

if __name__=="__main__":
    train_loader, val_loader, test_loader = load_dataset(
        video_path="../datasets/UCF50/",
        label_dir="../datasets/ucf50TrainTestlist/",
        video_dim=128, 
        chunk_length=30, 
        num_frames=15,
        batch_size=16
    )
    train_iter = iter(train_loader)
    x, y = next(train_iter)
    print("size of batch: ", x.size(), y.size())