import numpy as np
from torch.utils.data import DataLoader
from torch.utils.data.sampler import SubsetRandomSampler
import torchvision.transforms as transforms
from dataloader.gaussian_blur import GaussianBlur
from torchvision import datasets
from .dataset import ClrDataset

np.random.seed(0)


class DataSetWrapper(object):
    def __init__(self, 
                batch_size, 
                num_workers, 
                valid_size, 
                input_shape, 
                s, 
                csv_file,
                csv_test_file,
                img_root_dir,
                img_root_dir_test,
                img_path_col, 
                text_col, 
                text_from_files, 
                text_root_dir):
                
        self.batch_size = batch_size
        self.num_workers = num_workers
        self.valid_size = valid_size
        self.s = s
        self.input_shape = eval(input_shape)
        self.csv_file = csv_file
        self.csv_test_file = csv_test_file
        self.img_root_dir = img_root_dir
        self.img_root_dir_test = img_root_dir_test
        self.img_path_col = img_path_col 
        self.text_col = text_col
        self.text_from_files = text_from_files
        self.text_root_dir = text_root_dir

        
    def get_train_data_loaders(self):
        data_augment = self._get_simclr_pipeline_transform()
        train_dataset = ClrDataset(csv_file=self.csv_file,
                                    img_root_dir=self.img_root_dir,
                                    img_root_dir_test=self.img_root_dir_test,
                                    input_shape = self.input_shape,
                                    img_path_col = self.img_path_col, 
                                    text_col = self.text_col, 
                                    text_from_files = self.text_from_files, 
                                    text_root_dir = self.text_root_dir,
                                    mode = 'train',
                                    transform=SimCLRTrainDataTransform(data_augment)
                                    )
        print("num_train len : ", len(train_dataset))
        # print("train_dataset_data1")
        # print(train_dataset[0]['phrase'])
        # print("train_dataset_data2")
        # for ais, als in train_dataset:
        #     print(ais)

        train_loader, valid_loader = self.get_train_validation_data_loaders(train_dataset)
        return train_loader, valid_loader

    def get_test_data_loaders(self):
        data_augment = self._get_simclr_pipeline_transform()
        test_dataset = ClrDataset(csv_file=self.csv_test_file,
                                    img_root_dir=self.img_root_dir,
                                    img_root_dir_test=self.img_root_dir_test,
                                    input_shape = self.input_shape,
                                    img_path_col = self.img_path_col,
                                    text_col = self.text_col,
                                    text_from_files = self.text_from_files,
                                    text_root_dir = self.text_root_dir,
                                    mode='test',
                                    transform=SimCLRTestDataTransform(data_augment)
                                )

        print("num_test len : ", len(test_dataset))
        print(test_dataset)
        # print("train_dataset_data1")
        # print(train_dataset[0]['phrase'])
        # print("train_dataset_data2")

        # test_loader = DataLoader(test_dataset, batch_size=self.batch_size, num_workers=self.num_workers, shuffle=False)
        print("load test_loader....")

        return test_dataset

    def _get_simclr_pipeline_transform(self):
        # get a set of data augmentation transformations as described in the SimCLR paper.
        color_jitter = transforms.ColorJitter(0.8 * self.s, 0.8 * self.s, 0.8 * self.s, 0.2 * self.s)
        data_transforms = transforms.Compose([
                                              transforms.Scale((self.input_shape[0], self.input_shape[1])),
                                              transforms.RandomResizedCrop(size=self.input_shape[0], scale=(0.8, 1.0)),
                                              transforms.RandomHorizontalFlip(),
                                            #   transforms.RandomApply([color_jitter], p=0.8),
                                              transforms.RandomGrayscale(p=0.2),
                                            #   GaussianBlur(kernel_size=int(0.1 * self.input_shape[0])),
                                              transforms.ToTensor(),
                                              transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225)),
                                              ])
        return data_transforms

    def get_train_validation_data_loaders(self, train_dataset):
        # obtain training indices that will be used for validation
        num_train = len(train_dataset)
        indices = list(range(num_train))
        np.random.shuffle(indices)

        split = int(np.floor(self.valid_size * num_train))
        train_idx, valid_idx = indices[split:], indices[:split]

        # define samplers for obtaining training and validation batches
        train_sampler = SubsetRandomSampler(train_idx)
        valid_sampler = SubsetRandomSampler(valid_idx)

        # train_loader = DataLoader(train_dataset, batch_size=self.batch_size, num_workers=self.num_workers, drop_last=True, shuffle=False)
        # valid_loader = DataLoader(train_dataset, batch_size=self.batch_size, num_workers=self.num_workers, drop_last=True)

        train_loader = DataLoader(train_dataset, batch_size=self.batch_size, sampler=train_sampler,
                                  num_workers=self.num_workers, drop_last=True, shuffle=False)
        valid_loader = DataLoader(train_dataset, batch_size=self.batch_size, sampler=valid_sampler,
                                  num_workers=self.num_workers, drop_last=True)
        print("load train_loader....")
        # print(train_idx)
        # print(valid_idx)
        # print(len(train_loader))
        return train_loader, valid_loader

class SimCLRTrainDataTransform(object):
    def __init__(self, transform_image):
        self.transform_image = transform_image

    def __call__(self, sample):
        xi = self.transform_image(sample['image'])
        xl = sample['phrase']

        return xi, xl

class SimCLRTestDataTransform(object):
    def __init__(self, transform_image):
        self.transform_image = transform_image

    def __call__(self, sample):
        xi = self.transform_image(sample['image'])
        id = sample['img_id']

        return xi, id