import cProfile, time, math, os, shutil, numpy as np
from PIL import Image
# from bisect import insort

import matplotlib.pyplot as plt
plt.figure(figsize=(12, 4))

from manga_ocr import MangaOcr
ocr = MangaOcr()

colors_rgb = [(230, 25, 75), (60, 180, 75), (255, 225, 25), (67, 99, 216), (245, 130, 49), (145, 30, 180), (70, 240, 240), (240, 50, 230), (188, 246, 12), (250, 190, 190), (0, 128, 128), (230, 190, 255), (154, 99, 36), (255, 250, 200), (128, 0, 0), (170, 255, 195), (128, 128, 0), (255, 216, 177), (0, 0, 117), (128, 128, 128)]

masks_cache = {}

def circular_mask(radius):
	size = 2 * radius + 1
	y, x = np.ogrid[-radius:radius+1, -radius:radius+1]
	mask = (x*x + y*y) <= radius*radius
	return mask.astype(np.uint8)

def get_circular_mask(radius):
	if radius not in masks_cache:
		masks_cache[radius] = circular_mask(radius)
	return masks_cache[radius]

def label_groups(binary_image, radius=20):
	shape = binary_image.shape
	group_arr = np.full_like(binary_image, -1, dtype=int)
	group_arr[binary_image == 1] = -2
	group_id = 0
	outer_count = 4*math.floor(radius*math.sqrt(2))
	mask = get_circular_mask(radius)

	def help(start_x, start_y):

		def apply_mask(center_x, center_y):
			x1, x2 = max(center_x - radius, 0), min(center_x + radius + 1, shape[0])
			y1, y2 = max(center_y - radius, 0), min(center_y + radius + 1, shape[1])
			
			mx1, mx2 = x1 - (center_x - radius), radius + 1 + (x2 - (center_x + 1))
			my1, my2 = y1 - (center_y - radius), radius + 1 + (y2 - (center_y + 1))

			region = (group_arr[x1:x2, y1:y2] == -2) & (mask[mx1:mx2, my1:my2] == 1)
			coords = np.argwhere(region)
			coords[:, 0] += x1
			coords[:, 1] += y1
			return coords

		queue = [(start_x, start_y)]
		group_arr[start_x, start_y] = group_id

		while queue:
			cx, cy = queue.pop()
			outer_counter = 0
			for nx, ny in apply_mask(cx, cy):
				group_arr[nx, ny] = group_id
				if outer_counter < outer_count:
					queue.append((nx, ny))
				outer_counter += 1

	while np.any(group_arr == -2):
		# print(group_id)
		indices = np.transpose(np.where(group_arr == -2))
		x, y = indices[np.random.choice(len(indices))]
		help(x, y)
		group_id += 1
	
	return group_arr

def extract_boxes(group_arr, min_area=200):
	boxes = {}
	for y in range(group_arr.shape[0]):
		for x in range(group_arr.shape[1]):
			gid = int(group_arr[y, x])
			if gid != -1:
				if gid not in boxes:
					boxes[gid] = [x, x + 1, y, y + 1]
				else:
					boxes[gid][0] = min(boxes[gid][0], x)
					boxes[gid][1] = max(boxes[gid][1], x + 1)
					boxes[gid][2] = min(boxes[gid][2], y)
					boxes[gid][3] = max(boxes[gid][3], y + 1)
	
	return {gid: box for gid, box in boxes.items() if (box[1] - box[0]) * (box[3] - box[2]) > min_area}

def split_boxes(boxes, group_arr, col_averages, spacing_threshold=5, min_width=23):
	splitted = {}
	max_gid = max(boxes.keys())
	for gid, (x1, x2, y1, y2) in boxes.items():
		local_group_mask = np.where(group_arr == gid, 1, 0)
		col_avg = col_averages[x1:x2]

		zero_ranges = []
		in_zero = False
		start = 0
		for i, v in enumerate(col_avg):
			if v == 0 and not in_zero:
				in_zero = True
				start = i
			elif v != 0 and in_zero:
				if i - start >= spacing_threshold:
					zero_ranges.append((start, i))
				in_zero = False
		if in_zero and len(col_avg) - start >= spacing_threshold:
			zero_ranges.append((start, len(col_avg)))

		if not zero_ranges:
			splitted[gid] = (x1, x2, y1, y2)
			continue

		mids = [x1 + (start + end) // 2 for start, end in zero_ranges]
		split_xs = [x1] + mids + [x2]
		i = 0
		while i < len(split_xs) - 1:
			start = split_xs[i]
			end = split_xs[i + 1]
			width = end - start
			if width < min_width:
				split_xs.pop(i + 1)
				continue

			col_nonzero = np.nonzero(col_averages[start:end])[0]
			row_nonzero = np.nonzero(np.mean(local_group_mask[y1:y2, start:end], axis=1))[0]

			if col_nonzero.size == 0:
				print("impossible case reached")
				exit(1)

			new_x1 = int(start + col_nonzero[0])
			new_x2 = int(start + col_nonzero[-1] + 1)
			new_y1 = int(y1 + row_nonzero[0])
			new_y2 = int(y1 + row_nonzero[-1] + 1)

			max_gid += 1
			splitted[max_gid] = (new_x1, new_x2, new_y1, new_y2)
			i += 1

	return splitted

def merge_boxes(boxes):
	merged = []
	for gid, (x1, x2, y1, y2) in sorted(boxes.items(), key=lambda item: item[1][0]):
		for mgid, (mx1, mx2, my1, my2) in enumerate(merged):
			if not (x2 < mx1 or x1 > mx2):
				merged[mgid] = (min(x1, mx1),max(x2, mx2),min(y1, my1),max(y2, my2))
				break
		else:
			merged.append((x1, x2, y1, y2))
	return merged

def draw_rectangle_on_numpy_arr(img_arr, offsets, color=[255, 0, 0]):
	x1, x2, y1, y2 = offsets
	img_arr[y1:y2, x1] = color
	img_arr[y1:y2, x2 - 1] = color
	img_arr[y1, x1:x2] = color
	img_arr[y2 - 1, x1:x2] = color

def create_rgb_image(group_arr, colors_rgb, boxes):
	rgb_arr = np.ones((*group_arr.shape, 3), dtype=np.uint8) * 255
	for box_index, (x1, x2, y1, y2) in enumerate(boxes):
		rgb_arr[y1:y2, x1:x2][group_arr[y1:y2, x1:x2] != -1] = colors_rgb[box_index % len(colors_rgb)]
		draw_rectangle_on_numpy_arr(rgb_arr, (x1, x2, y1, y2))
	return Image.fromarray(rgb_arr)

def process(chapter, page):
	page_folder = f"./{chapter}/{page}"
	groups_folder = f"{page_folder}/groups"
	
	if os.path.exists(page_folder):
		shutil.rmtree(page_folder)
	os.makedirs(groups_folder)

	arr = np.array(Image.open(f"{page_folder}.png").convert("L"))
	binary_image = (arr != 255).astype(np.uint8)

	# start_time = time.time()

	group_arr = label_groups(binary_image)
	col_averages = np.mean(np.where(group_arr != -1, 1, 0), axis=0)
	raw_boxes = extract_boxes(group_arr)
	boxes = []
	if len(raw_boxes) > 0:
		splitted_boxes = split_boxes(raw_boxes, group_arr, col_averages)
		boxes = merge_boxes(splitted_boxes)
		boxes = sorted(boxes, key=lambda b: -b[0])

	# print(time.time() - start_time)

	for gid, (x1, x2, y1, y2) in enumerate(boxes):
		# print(gid, x2 - x1, y2 - y1)
		plt.clf()
		col_avg = col_averages[x1:x2]
		plt.plot(col_avg)
		plt.axhline(y=col_avg.mean(), color='red')
		plt.savefig(f"{groups_folder}/{gid:02}_plot.png")

	create_rgb_image(group_arr, colors_rgb, boxes).save(f"{page_folder}/groups.png")

	for gid, (x1, x2, y1, y2) in enumerate(boxes):
		crop_img = Image.fromarray(np.pad(arr[y1:y2, x1:x2], ((15, 15), (15, 15)), mode='constant', constant_values=255))
		crop_img.save(f"{groups_folder}/{gid:02}.png")
		text = ocr(crop_img)
		with open(f"{groups_folder}/{gid:02}.txt", "w", encoding="utf-8") as f:
			f.write(text)

def main():
	for chapter in range(1, 5+1):
		for page_filename in os.listdir(f"./{chapter}"):
			if not page_filename.endswith(".png"):
				continue
			page = page_filename.split(".")[0]

			if int(page) < 8:
				continue

			st = time.time()
			process(chapter, page)
			print(chapter, page, time.time() - st)

# cProfile.run('main()')

main()
# process(1, 97)
# process(1, 111)
# process(1, 117)
# process(1, 128)