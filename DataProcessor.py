import os
from os.path import join
import cv2
import random
import shutil


def randomSortFrame(frameDir, percentage=0.2):
    saveDir = join(frameDir, os.pardir, "Sorted")
    os.makedirs(saveDir, exist_ok=True)
    frames = os.listdir(frameDir)
    random.shuffle(frames)
    frames = frames[:int(len(frames)*percentage)]
    for frame in frames:
        shutil.copy(join(frameDir, frame), join(saveDir, frame))


if __name__ == '__main__':
    randomSortFrame("data/Frames")
