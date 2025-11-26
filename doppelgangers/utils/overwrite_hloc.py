import h5py
from tqdm import tqdm



def main(sfm_filtered, pair_path, matches_file, threshold=0.8):
    with open(sfm_filtered, 'r', encoding='utf-8') as filtered_f, open(pair_path, 'w', encoding='utf-8') as orig_f,  h5py.File(matches_file, 'r+') as matches_f:
        total = 0
        filtered = 0
        for l in tqdm(filtered_f):
            total += 1
            i1, i2, good_pair_prob = l.split()
            if float(good_pair_prob) < threshold:
                del matches_f[i1][i2]
                filtered += 1
            else:
                orig_f.write(f"{i1} {i2} \n")
        print(f"Doppelgangers Filtered {filtered / total * 100 :.2f}%")
