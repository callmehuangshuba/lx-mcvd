# https://github.com/edenton/svg/blob/master/data/kth.py
import numpy as np
import os
import pickle
import torch
import h5py
from PIL import Image
from torch.utils.data import Dataset
from torchvision import transforms

from .h5 import HDF5Dataset
from .npy import TCG_NPYDataset
import torch.nn.functional as F
from concurrent.futures import ThreadPoolExecutor

class TCGVideoDataset(Dataset):
    def __init__(self, 
        time_resolution,
        data_dir, 
        type='train',
        type2=None,
        type3=None,
        total_videos=-1, 
        num_frames=40, 
        image_size=64, 
        random_time=True,
        # color_jitter=None, 
        random_horizontal_flip=False
    ):
        super(TCGVideoDataset, self).__init__()
        self.time_resolution=time_resolution
        self.data_dir = data_dir
        self.type = type
        self.type2 = type2
        self.type2 = type3
        self.num_frames = num_frames
        self.image_size = image_size
        self.total_videos = total_videos
        self.random_time = random_time
        self.random_horizontal_flip = random_horizontal_flip
        # self.jitter = transforms.ColorJitter(hue=color_jitter) if color_jitter else None
        
        self.videos_ds = TCG_NPYDataset(os.path.join(self.data_dir, type),os.path.join(self.data_dir, type2),os.path.join(self.data_dir, type3))
        
    def __len__(self):
        if self.total_videos > 0:
            return self.total_videos
        else:
            if "UCF" in self.data_dir:
                return self.num_train_vids if self.type=='train' else self.num_test_vids
            else:
                return len(self.videos_ds)

    def len_of_vid(self, index):
        video_index = index % self.__len__()
        shard_idx, idx_in_shard = self.videos_ds.get_indices(video_index)
        with self.videos_ds.opener(self.videos_ds.shard_paths[shard_idx]) as f:
            video_len = f['len'][str(idx_in_shard)][()]
        return video_len

    def max_index(self):
        if "UCF" in self.data_dir:
            return self.num_train_vids if self.type=='train' else self.num_test_vids
        else:  
            return len(self.videos_ds)
    def read_frames(self,frame_name):
        with h5py.File(frame_name, 'r') as file: 
            arr=file['Infrared'][:]
            # arr=file['b14'][:]
            # arr=(frame-frame.min())/(frame.max()-frame.min())
            arr = F.interpolate(torch.from_numpy(arr).unsqueeze(0).unsqueeze(0), size=(arr.shape[0]//4, arr.shape[1]//4), mode='bilinear', align_corners=False).squeeze(0).squeeze(0)
            return arr
    
    def __getitem__(self, index, time_idx=0):
        
        video_index = round(index / (self.__len__() - 1) * (self.max_index() - 1))
        # shard_idx, idx_in_shard = self.videos_ds.get_indices(video_index)
        one_TC_vid_path = self.videos_ds.total_TC_vid_list[video_index]
        one_TC_vid_path_era5 = self.videos_ds.total_era5_vid_list[video_index]
        one_TC_vid_path_era5_3d = self.videos_ds.total_era5_3d_vid_list[video_index]
        # one_TC_h5_list=self.videos_ds.total_TC_vid_list[video_index]
        prefinals = []
        prefinals2 = []
        prefinals3 = []
        num_frames=len(one_TC_vid_path)
        # num_frames=len(one_TC_h5_list)
        # sample frames
        if self.random_time and num_frames > self.num_frames:
            # sampling start frames
            time_idx = np.random.choice(num_frames - (self.num_frames-1)*self.time_resolution) # 47
        # time_idx=0
        prefinals = one_TC_vid_path[time_idx:min(time_idx + self.num_frames * self.time_resolution, num_frames):self.time_resolution]
        prefinals2 = one_TC_vid_path_era5[time_idx:min(time_idx + self.num_frames * self.time_resolution, num_frames):self.time_resolution]
        prefinals3 = one_TC_vid_path_era5_3d[time_idx:min(time_idx + self.num_frames * self.time_resolution, num_frames):self.time_resolution]

        data = torch.stack(prefinals)
        min_=data.min()
        max_=data.max()
        data=(data-data.min())/(data.max()-data.min())

        data2 = torch.stack(prefinals2)
        min_sst = np.nanmin(data2[:,0].cpu().numpy())
        max_sst = np.nanmax(data2[:,0].cpu().numpy())
        min_2 = data2.amin(dim=(0, 2, 3), keepdim=True)
        max_2 = data2.amax(dim=(0, 2, 3), keepdim=True)
        min_2[0][0]=torch.tensor(min_sst, dtype=min_2.dtype, device=min_2.device)
        max_2[0][0]=torch.tensor(max_sst, dtype=min_2.dtype, device=min_2.device)
        data2 = (data2-min_2)/(max_2-min_2)



        data3 = torch.stack(prefinals3)
        min_3 = data3.amin(dim=(0,  3,4), keepdim=True)
        max_3 = data3.amax(dim=(0,  3,4), keepdim=True)
        data3 = (data3-min_3)/(max_3-min_3)
        # return data, video_index,{"min":min_,"max":max_}
        return (data,data2,data3), video_index


class KTHDataset(Dataset):

    def __init__(self, data_dir, frames_per_sample=5, train=True, random_time=True, random_horizontal_flip=True,
                 total_videos=-1, with_target=True, start_at=0):
        if train:
            self.data_dir = os.path.join(data_dir,'train+val')     
        else:
            self.data_dir = os.path.join(data_dir,'test')                  # '/path/to/Datasets/KTH64_h5' (with shard_0001.hdf5 and persons.pkl in it)
        self.train = train
        self.frames_per_sample = frames_per_sample
        self.random_time = random_time
        self.random_horizontal_flip = random_horizontal_flip
        self.total_videos = total_videos            # If we wish to restrict total number of videos (e.g. for val)
        self.with_target = with_target
        self.start_at = start_at

        # Read h5 files as dataset
        self.videos_ds = TCG_HDF5Dataset(self.data_dir)
        '''
        # Persons
        with open(os.path.join(data_dir, 'persons.pkl'), 'rb') as f:
            self.persons = pickle.load(f)

        # Train
        self.train_persons = list(range(1, 21))
        self.train_idx = sum([self.persons[p] for p in self.train_persons], [])
        # Test
        self.test_persons = list(range(21, 26))
        self.test_idx = sum([self.persons[p] for p in self.test_persons], [])
        '''
        print(f"Dataset length: {self.__len__()}")

    def len_of_vid(self, index):
        video_index = index % self.__len__()
        shard_idx, idx_in_shard = self.videos_ds.get_indices(video_index)
        with self.videos_ds.opener(self.videos_ds.shard_paths[shard_idx]) as f:
            video_len = f['len'][str(idx_in_shard)][()]
        return video_len

    def __len__(self):
        return self.total_videos if self.total_videos > 0 else len(self.train_idx) if self.train else len(self.test_idx)

    def max_index(self):
        return len(self.train_idx) if self.train else len(self.test_idx)

    def __getitem__(self, index, time_idx=0):

        # Use `index` to select the video, and then
        # randomly choose a `frames_per_sample` window of frames in the video
        video_index = round(index / (self.__len__() - 1) * (self.max_index() - 1))
        shard_idx, idx_in_shard = self.videos_ds.get_indices(video_index)
        idx = self.train_idx[int(idx_in_shard)] if self.train else self.test_idx[int(idx_in_shard)]

        prefinals = []
        flip_p = np.random.randint(2) == 0 if self.random_horizontal_flip else 0
        with self.videos_ds.opener(self.videos_ds.shard_paths[shard_idx]) as f:
            video_len = f['len'][str(idx)][()] - self.start_at
            if self.random_time and video_len > self.frames_per_sample:
                time_idx = np.random.choice(video_len - self.frames_per_sample)
            time_idx += self.start_at
            for i in range(time_idx, min(time_idx + self.frames_per_sample, video_len)):
                img = f[str(idx)][str(i)][()]
                arr = transforms.RandomHorizontalFlip(flip_p)(transforms.ToTensor()(img))
                prefinals.append(arr)
            target = int(f['target'][str(idx)][()])

        if self.with_target:
            return torch.stack(prefinals), torch.tensor(target)
        else:
            return torch.stack(prefinals)
