import torch
from torch.utils.data import Dataset, DataLoader
from torchvision.transforms import functional as F
import json
import random
import os
import cv2


class FishDataset(Dataset):
    def __init__(self, img_dir, lab_path, transforms, keypoint_names=("Head", "Dorsal1", "Dorsal2", "Pectoral", "Gluteal", "Caudal")):
        self.img_dir = img_dir
        with open(lab_path, "r") as f:
            lab = json.load(f)
        lab = [v for k, v in lab.items() if
               len([reg for reg in v["regions"] if reg["shape_attributes"]["name"] == "rect"]) > 0]
        self.imgs = []
        self.boxess = []
        self.keypointsss = []
        for e in lab:
            img = e["filename"]
            self.imgs.append(img)
            boxes = [[region["shape_attributes"]["x"],
                      region["shape_attributes"]["y"],
                      region["shape_attributes"]["x"] + region["shape_attributes"]["width"],
                      region["shape_attributes"]["y"] + region["shape_attributes"]["height"]]
                     for region in e["regions"] if region["shape_attributes"]["name"] == "rect"]
            self.boxess.append(boxes)
            keypointss = []
            for box in boxes:
                xmin, ymin, xmax, ymax = box
                keypoints = [[0, 0, 0] for k in keypoint_names]
                for i, k in enumerate(keypoint_names):
                    for region in e["regions"]:
                        if region["shape_attributes"]["name"] == "point":
                            key = region["region_attributes"]["Keypoint"]
                            x, y = region["shape_attributes"]["cx"], region["shape_attributes"]["cy"]
                            if key == k and xmin <= x <= xmax and ymin <= y <= ymax:
                                keypoints[i] = [x, y, 1]  # 1 means visibility = True
                keypointss.append(keypoints)
            self.keypointsss.append(keypointss)
        self.transforms = transforms

    def __getitem__(self, idx):
        img_path = os.path.join(self.img_dir, self.imgs[idx])
        image = cv2.imread(img_path)
        b, g, r = cv2.split(image)
        image = cv2.merge([r, g, b])
        boxes = self.boxess[idx].copy()
        keypointss = self.keypointsss[idx].copy()
        labels = torch.ones((len(boxes),), dtype=torch.int64)  # Only have one class
        boxes = torch.as_tensor(boxes, dtype=torch.float32)
        keypointss = torch.as_tensor(keypointss, dtype=torch.float32)
        image_id = torch.tensor([idx])
        target = {}
        target["boxes"] = boxes
        target["labels"] = labels
        target["keypoints"] = keypointss
        target["image_id"] = image_id
        if self.transforms is not None:
            image, target = self.transforms(image, target)
        return image, target

    def __len__(self):
        return len(self.imgs)


class ToTensor(object):
    def __call__(self, image, target):
        image = F.to_tensor(image)
        return image, target


class RandomHorizontalFlip(object):
    def __init__(self, prob):
        self.prob = prob

    def __call__(self, image, target):
        if random.random() < self.prob:
            height, width = image.shape[-2:]
            image = image.flip(-1)

            bbox = target["boxes"]
            bbox[:, [0, 2]] = width - bbox[:, [2, 0]]
            target["boxes"] = bbox
            keypoints = target["keypoints"]
            for keypoint in keypoints:
                for kp in keypoint:
                    kp[0] = width - kp[0]
            target["keypoints"] = keypoints
        return image, target


class Compose(object):
    def __init__(self, transforms):
        self.transforms = transforms

    def __call__(self, image, target):
        for t in self.transforms:
            image, target = t(image, target)
        return image, target


def get_transforms(train):
    transforms = []
    transforms.append(ToTensor())
    if train:
        transforms.append(RandomHorizontalFlip(0.5))
    return Compose(transforms)


def collate_fn(batch):
    return tuple(zip(*batch))  # Default collate_fn needs same size of images, so should re-define it here


def get_dataloader(train, batch_size, val_split=0.0, shuffle=True, num_workers=0):
    dataset = FishDataset("data/Frames", "data/Annotation/annotation_keypoint.json", get_transforms(train=train))
    if val_split == 0.0:
        dataset_train = dataset
        dataloader_train = DataLoader(dataset_train, batch_size=batch_size, shuffle=shuffle, num_workers=num_workers, collate_fn=collate_fn)
        dataloader_val = None
    else:
        indices = torch.randperm(len(dataset)).tolist()
        where_train = int(len(dataset) * (1 - val_split))
        dataset_train = torch.utils.data.Subset(dataset, indices[:where_train])
        dataset_val = torch.utils.data.Subset(dataset, indices[where_train:])
        dataloader_train = DataLoader(dataset_train, batch_size=batch_size, shuffle=shuffle, num_workers=num_workers, collate_fn=collate_fn)
        dataloader_val = DataLoader(dataset_val, batch_size=batch_size, shuffle=shuffle, num_workers=num_workers, collate_fn=collate_fn)
    return dataloader_train, dataloader_val


if __name__ == '__main__':
    dataset = FishDataset("data/Frames", "data/Annotation/annotation_keypoint.json", get_transforms(train=True))
    a = next(iter(dataset))
    pass
