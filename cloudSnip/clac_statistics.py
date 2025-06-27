from torchgeo.datasets import RasterDataset
import rasterio as rio
from dataset import train_img
import numpy as np
from pathlib import Path

def calc_statistics(dset: RasterDataset):
    """
    Calculate the statistics (mean and std) for the entire dataset
    Warning: This is an approximation. The correct value should take into account the
    mean for the whole dataset for computing individual stds.
    For correctness I suggest checking: http://notmatthancock.github.io/2017/03/23/simple-batch-stat-updates.html
    """

    # To avoid loading the entire dataset in memory, we will loop through each img
    # The filenames will be retrieved from the dataset's rtree index
    # files = [
    #     item.object for item in dset.index.intersection(dset.index.bounds, objects=True)
    # ]
    path = Path("/home/nischal/projects/cloudSnip_data/processed")
    files = list(path.glob("*.tif"))
    print(files)
    # print(dset.index.bounds)

    print(files)
    # Resetting statistics
    accum_mean = 0
    accum_std = 0

    for file in files:
        # img = rio.open(file).read() / 10000  # type: ignore
        img = rio.open(file).read()
        accum_mean += np.nanmean(img.reshape((img.shape[0], -1)), axis=1)
        accum_std += np.nanstd(img.reshape((img.shape[0], -1)), axis=1)

    # at the end, we shall have 2 vectors with length n=chnls
    # we will average them considering the number of images
    return accum_mean / len(files), accum_std / len(files)

if __name__ == "__main__":
    mean, std = calc_statistics(train_img)
    print(f"Mean: {mean}, Std: {std}")
    # Save the statistics to a file
    with open("statistics.txt", "w") as f:
        f.write(f"Mean: {mean.tolist()}\n")
        f.write(f"Std: {std.tolist()}\n")