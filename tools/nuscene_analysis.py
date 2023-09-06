from nuscenes.nuscenes import NuScenes
import numpy as np

def process():
    # version='v1.0-trainval', dataroot='/data/nuscenes/full', verbose=True
    nusc = NuScenes(version='v1.0-mini', dataroot='/data/nuscenes/mini', verbose=True)
    samples = nusc.sample

    for sample in samples[:1]:
        sample.timestamp
        sample.anns


process()