import numpy as np
from scipy.io import loadmat
from utils import *

def main():
    # Load Matches
    data = loadmat('data/tracks.mat')
    track_x = data['track_x']
    track_y = data['track_y']

    M, S = affineSFM(track_x,track_y)

    plotShape(S)
    plotCameraMotion(M)

if __name__=='__main__':
    main()