from types import SimpleNamespace

from tqdm import tqdm
from ..models.cnn_classifier import decoder
from ..datasets.hloc_dataset import HlocDoppelgangersDataset
import torch
from torch.utils.data import DataLoader
import copy
import h5py
import argparse
from scipy.special import softmax



def get_args():
    # command line args
    parser = argparse.ArgumentParser(description='Structure from Motion disambiguation with Doppelgangers classification model.')
    
    # colmap setting
    parser.add_argument('--weights_path', type=str,
                        help="Path to classifier weights")
    parser.add_argument('--matches_path', type=str,
                        help="path to hloc matches HDF5 file")
    parser.add_argument('--filtered_path', type=str,
                        help="path to hloc matches HDF5 file")
    parser.add_argument('--image_dir', type=str,
                        help="path to where images are stored")
    parser.add_argument('--batch_size', type=int, default=16,
                        help="batch size")

    return parser.parse_args()

def main(
    weights_path,
    matches_file,
    sfm_filtered,
    image_dir,
    batch_size,
):
    model = decoder(cfg=SimpleNamespace(input_dim=10))
    ckpt = torch.load(weights_path)
    new_ckpt = copy.deepcopy(ckpt['dec'])

    for key, _ in ckpt['dec'].items():
        if 'module.' in key:
            new_ckpt[key[len('module.'):]] = new_ckpt.pop(key)

    model.load_state_dict(new_ckpt, strict=True)
    model = model.cuda().eval()

    with h5py.File(matches_file, 'r') as matches_f,  open(sfm_filtered, 'w', encoding='utf-8') as filterd_f:
        test_loader = DataLoader(
            dataset=HlocDoppelgangersDataset(
                img_size=640,
                image_dir=image_dir,
                matches_file=matches_f
            ),
            batch_size=batch_size,
            shuffle=False, num_workers=8, drop_last=False)

        for b in tqdm(test_loader):
            with torch.no_grad():
                scores = model(b['image'].cuda()).detach().cpu().numpy()
                for i1, i2, score in zip(b['image1_name'], b['image2_name'], scores):
                    good_pair_prob = softmax(score)[1]
                    filterd_f.write(f"{i1} {i2} {good_pair_prob}\n")



if __name__ == "__main__":
    args = get_args()

    main(
    weights_path=args.weights_path,
    matches_file=args.matches_path,
    sfm_filtered=args.filtered_path,
    image_dir=args.image_dir,
    batch_size=args.batch_size,
    )
