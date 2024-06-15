from ex2_utils import *
import matplotlib.pyplot as plt
import time


def main():
    conv1Demo()
    conv2Demo()
    derivDemo()
    blurDemo('./boxman.jpeg', 9)
    edgeDemo('./boxman.jpeg', 0.7,50,150)
    # houghDemo()


if __name__ == '__main__':
    main()
