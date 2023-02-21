import numpy as np

wczytaj = np.genfromtxt("car.txt", dtype=str)
print(wczytaj)
print(wczytaj.shape[0])