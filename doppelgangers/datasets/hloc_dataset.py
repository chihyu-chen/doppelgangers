import os.path as osp
import numpy as np
import torch
from torch.utils.data import Dataset
import cv2
from ..utils.dataset import read_loftr_matches



class HlocDoppelgangersDataset(Dataset):
    def __init__(self,
                 image_dir,
                 features_file,
                 matches_file,
                 pair_path,
                 img_size,
                 **kwargs):
        """
        Doppelgangers test dataset: loading images and loftr matches for Doppelgangers model.
        
        Args:
            image_dir (str): root directory for images.
            loftr_match_dir (str): root directory for loftr matches.
            pair_path (str): pair_list.npy path. This contains image pair information.
            img_resize (int, optional): the longer edge of resized images. None for no resize. 640 is recommended.
                                        This is useful during training with batches and testing with memory intensive algorithms.
        """
        super().__init__()
        self.image_dir = image_dir    
        self.features_f = features_file
        self.matches_f = matches_file
        self.pairs_info = []
        for i1 in matches_file.keys():
            for i2 in matches_file[i1].keys():
                self.pairs_info.append((i1, i2))

        self.img_size = img_size

        
    def __len__(self):
        return len(self.pairs_info)

    def __getitem__(self, idx):
        image_1_name, image_2_name = self.pairs_info[idx]
        
        keypoints1_f = np.asarray(self.features_f[image_1_name]['keypoints'])
        keypoints2_f = np.asarray(self.features_f[image_2_name]['keypoints'])
        print(f"idx {idx}: keypoints1 shape from features: {keypoints1_f.shape}, keypoints2 shape from features: {keypoints2_f.shape}")
        matches_data = self.matches_f[image_1_name][image_2_name]
        keypoints1_m = np.array(matches_data['keypoints0'])
        keypoints2_m = np.array(matches_data['keypoints1'])
        print(f"idx {idx}: keypoints1 shape from matches: {keypoints1_m.shape}, keypoints2 shape from matches: {keypoints2_m.shape}")
        matches = np.array(matches_data['matches0'])
        actual_matches = matches > -1
        matches = matches[actual_matches]
        print(f"idx {idx}: max of matches: {matches.max()}")
        conf = np.array(matches_data['matching_scores0'])
        conf = conf[actual_matches]
        keypoints1 = keypoints1_f[matches].astype(np.int32)
        keypoints2 = keypoints2_f[matches].astype(np.int32)

        if np.sum(conf>0.8) == 0:
            matches = None
        else:
            F, mask = cv2.findFundamentalMat(keypoints1[conf>0.8],keypoints2[conf>0.8],cv2.FM_RANSAC, 3, 0.99)
            if mask is None or F is None:
                matches = None
            else:
                matches = np.array(np.ones((keypoints1.shape[0], 2)) * np.arange(keypoints1.shape[0]).reshape(-1,1)).astype(int)[conf>0.8][mask.ravel()==1]

        img_name1 = osp.join(self.image_dir, image_1_name)
        img_name2 = osp.join(self.image_dir, image_2_name)

        image = read_loftr_matches(img_name1, img_name2, self.img_size, 8, True, keypoints1, keypoints2, matches, warp=True, conf=conf)
        
        return {
            'image': image,
            'image1_name': image_1_name,
            'image2_name': image_2_name,
        }

def get_datasets(cfg):
    te_dataset = HlocDoppelgangersDataset(
                    cfg.image_dir,
                    cfg.features_file,
                    cfg.matches_file,
                    cfg.test.pair_path,
                    img_size=getattr(cfg.test, "img_size", 640))

    return te_dataset


def init_np_seed(worker_id):
    seed = torch.initial_seed()
    np.random.seed(seed % 4294967296)


def get_data_loaders(cfg):
    te_dataset = get_datasets(cfg)    
    test_loader = torch.utils.data.DataLoader(
        dataset=te_dataset, batch_size=cfg.test.batch_size,
        shuffle=False, num_workers=cfg.num_workers, drop_last=False,
        worker_init_fn=init_np_seed)

    loaders = {
        "test_loader": test_loader,
    }
    return loaders
