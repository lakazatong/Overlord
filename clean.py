import os, math
from PIL import Image
import numpy as np
import matplotlib.pyplot as plt
from collections import defaultdict

def bezier_curve(t, P0, P1, P2, P3):
	x = (1 - t)**3 * P0[0] + 3 * (1 - t)**2 * t * P1[0] + 3 * (1 - t) * t**2 * P2[0] + t**3 * P3[0]
	y = (1 - t)**3 * P0[1] + 3 * (1 - t)**2 * t * P1[1] + 3 * (1 - t) * t**2 * P2[1] + t**3 * P3[1]
	return x, y

def generate_curve(P0, P1, P2, P3):
	arr = [-1] * 256
	x0, y0 = bezier_curve(0.0, P0, P1, P2, P3)
	arr[x0] = y0
	x1, y1 = bezier_curve(1.0, P0, P1, P2, P3)
	arr[x1] = y1
	recursive_curve(0.0, 1.0, P0, P1, P2, P3, arr)
	return arr

P0 = (0, 0)
P1 = (256,128)
P2 = (128,256)
P3 = (256,256)

bins = defaultdict(list)
for x, y in [bezier_curve(t, P0, P1, P2, P3) for t in np.arange(0, 1, 0.00001)]:
	bins[int(round(x))].append(y)

table = [255 if x >= 250 else min(255, int(round(np.mean(bins[x])))) for x in range(256)]

plt.scatter(range(256), table, s=1)
plt.savefig('curve.png')

# print(table)

# exit(0)

def replace_extension(path, new_ext):
	base, _ = os.path.splitext(path)
	return base + new_ext

def process_folder(folder_path):
	# i = 0
	# goto = 8
	for filename in os.listdir(folder_path):
		if filename.lower().endswith(('.jpg', '.jpeg')):
			# if i < goto - 1:
			# 	i += 1
			# 	continue
			print(f"Processing ./{folder_path}/{filename}...", end="")
			
			image_path = os.path.join(folder_path, filename)
			Image.fromarray(np.array(Image.open(image_path).convert('L').point(lambda p: table[p]))[:-200]).save(replace_extension(image_path, ".png"))
			
			print(" done")
			# break

for folder in ["1", "2", "3", "4", "5"]:
	process_folder(folder)
