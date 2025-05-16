import cProfile, time, math, os, shutil, numpy as np
from PIL import Image

import matplotlib.pyplot as plt
plt.figure(figsize=(12, 4))

from manga_ocr import MangaOcr
ocr = MangaOcr()

from utils import *

colors_rgb = [(230, 25, 75), (60, 180, 75), (255, 225, 25), (67, 99, 216), (245, 130, 49), (145, 30, 180), (70, 240, 240), (240, 50, 230), (188, 246, 12), (250, 190, 190), (0, 128, 128), (230, 190, 255), (154, 99, 36), (255, 250, 200), (128, 0, 0), (170, 255, 195), (128, 128, 0), (255, 216, 177), (0, 0, 117), (128, 128, 128)]

def label_groups(white_on_black_normalized, radius=15):
	shape = white_on_black_normalized.shape
	group_arr = np.full_like(white_on_black_normalized, -1, dtype=int)
	group_arr[white_on_black_normalized != 0] = -2
	gid = 0
	mask = get_circular_mask(radius)

	def help(start_x, start_y):

		def apply_mask(center_x, center_y):
			x1, x2 = max(center_x - radius, 0), min(center_x + radius + 1, shape[0])
			y1, y2 = max(center_y - radius, 0), min(center_y + radius + 1, shape[1])
			
			mx1, mx2 = max(radius - center_x, 0), min(radius + x2 - center_x, mask.shape[0])
			my1, my2 = max(radius - center_y, 0), min(radius + y2 - center_y, mask.shape[1])

			region = (group_arr[x1:x2, y1:y2] == -2) & (mask[mx1:mx2, my1:my2] == 1)
			coords = np.argwhere(region)
			coords[:, 0] += x1
			coords[:, 1] += y1
			return coords

		queue = [(start_x, start_y)]
		group_arr[start_x, start_y] = gid

		while queue:
			cx, cy = queue.pop()
			for nx, ny in apply_mask(cx, cy):
				group_arr[nx, ny] = gid
				queue.append((nx, ny))

	while np.any(group_arr == -2):
		indices = np.transpose(np.where(group_arr == -2))
		x, y = indices[np.random.choice(len(indices))]
		help(x, y)
		gid += 1
	
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
	
	return [(gid, tuple(box)) for gid, box in sorted(boxes.items()) if (box[1] - box[0]) * (box[3] - box[2]) > min_area]

def get_zero_ranges(group_mask, offsets, spacing_threshold, axis, zero_threshold=0):
	x1, x2, y1, y2 = offsets
	avg = np.mean(group_mask[y1:y2, x1:x2], axis=axis)

	binary = [0 if v <= zero_threshold else 1 for v in avg]

	in_zero = binary[0] == 0
	start = 0
	binary.pop(0)

	zero_ranges = []
	i = 1

	while binary:
		curr = binary.pop(0)
		if curr == 0 and not in_zero:
			in_zero = True
			start = i
		elif curr == 1 and in_zero:
			if i - start >= spacing_threshold:
				zero_ranges.append((start, i))
			in_zero = False
		i += 1

	if in_zero and len(avg) - start >= spacing_threshold:
		zero_ranges.append((start, len(avg)))

	return zero_ranges

def split_on_column(boxes, white_on_black_normalized, group_arr, spacing_threshold=5, min_width=23):
	splitted = []
	max_gid = max(boxes, key=lambda x: x[0])[0]
	for gid, offsets in boxes:
		group_mask = np.where(group_arr == gid, white_on_black_normalized, 0)
		zero_ranges = get_zero_ranges(group_mask, offsets, spacing_threshold, 0)
		if len(zero_ranges) == 0:
			splitted.append((gid, offsets))
			continue
		
		x1, x2, y1, y2 = offsets
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

			col_nonzero = np.nonzero(np.mean(group_mask[y1:y2, start:end], axis=0))[0]
			row_nonzero = np.nonzero(np.mean(group_mask[y1:y2, start:end], axis=1))[0]

			if col_nonzero.size == 0 or row_nonzero.size == 0:
				print("split_on_column: impossible case reached")
				print(y1, y2)
				print(start, end)
				exit(1)

			new_x1 = int(start + col_nonzero[0])
			new_x2 = int(start + col_nonzero[-1] + 1)
			new_y1 = int(y1 + row_nonzero[0])
			new_y2 = int(y1 + row_nonzero[-1] + 1)

			max_gid += 1
			splitted.append((max_gid, (new_x1, new_x2, new_y1, new_y2)))
			i += 1

	return splitted

def merge_boxes(boxes, overlap_threshold=0.5):
	def should_merge(x1, x2, mx1, mx2):
		overlap_x = max(0, min(x2, mx2) - max(x1, mx1))
		return max(overlap_x / float(x2 - x1), overlap_x / float(mx2 - mx1)) >= overlap_threshold

	merged = True
	while merged:
		merged = False
		i = 0
		while i < len(boxes):
			j = 0
			while j < len(boxes):
				if i == j:
					j += 1
					continue
				gid, (x1, x2, y1, y2) = boxes[i]
				_, (mx1, mx2, my1, my2) = boxes[j]
				if should_merge(x1, x2, mx1, mx2):
					boxes[i] = (gid, (min(x1, mx1), max(x2, mx2), min(y1, my1), max(y2, my2)))
					boxes.pop(j)
					merged = True
				else:
					j += 1
			i += 1
	return boxes

def split_on_ponctuation(boxes, white_on_black_normalized, group_arr, spacing_threshold=15, min_height=30):
	gid = 0
	splitted = []
	for old_gid, offsets in boxes:
		group_mask = np.where(group_arr == old_gid, white_on_black_normalized, 0)
		zero_ranges = get_zero_ranges(group_mask, offsets, spacing_threshold, 1)
		if len(zero_ranges) == 0:
			splitted.append((gid, offsets))
			gid += 1
			continue
		
		x1, x2, y1, y2 = offsets
		col = []
		mids = [y1 + (start + end) // 2 for start, end in zero_ranges]
		split_ys = [y1] + mids + [y2]

		i = 0
		while i < len(split_ys) - 1:
			if split_ys[i + 1] - split_ys[i] < min_height:
				del split_ys[i:i+2]
				if i > 0:
					i -= 1
			else:
				i += 1

		for i in range(len(split_ys) - 1):
			start, end = split_ys[i], split_ys[i + 1]

			col_nonzero = np.nonzero(np.mean(group_mask[start:end, x1:x2], axis=0))[0]
			row_nonzero = np.nonzero(np.mean(group_mask[start:end, x1:x2], axis=1))[0]

			if col_nonzero.size == 0 or row_nonzero.size == 0:
				print("split_on_ponctuation: impossible case reached")
				print(start, end)
				print(x1, x2)
				exit(1)

			new_x1 = int(x1 + col_nonzero[0])
			new_x2 = int(x1 + col_nonzero[-1] + 1)
			new_y1 = int(start + row_nonzero[0])
			new_y2 = int(start + row_nonzero[-1] + 1)

			splitted.append((gid, (new_x1, new_x2, new_y1, new_y2)))
			gid += 1

	return splitted

def split_on_chooonpu(boxes, white_on_black_normalized, group_arr, spacing_threshold=50, min_height=30):
	gid = 0
	splitted = []
	for old_gid, offsets in boxes:
		group_mask = np.where(group_arr == old_gid, white_on_black_normalized, 0)
		zero_ranges = get_zero_ranges(group_mask, offsets, spacing_threshold, 1, 0.1)
		if len(zero_ranges) == 0:
			splitted.append((gid, offsets))
			gid += 1
			continue
		
		x1, x2, y1, y2 = offsets
		col = []
		mids = [y1 + (start + end) // 2 for start, end in zero_ranges]
		split_ys = [y1] + mids + [y2]

		n = len(split_ys)

		def help():
			nonlocal gid
			start, end = split_ys[0], split_ys[1]

			col_mean = np.mean(group_mask[start:end, x1:x2], axis=0)
			row_mean = np.mean(group_mask[start:end, x1:x2], axis=1)
			col_nonzero = np.nonzero(col_mean)[0]
			row_zero = np.where(row_mean == 0)[0]
			row_nonzero = np.nonzero(row_mean)[0]

			if col_nonzero.size != 0 and row_zero.size != 0:
				new_x1 = int(x1 + col_nonzero[0])
				new_x2 = int(x1 + col_nonzero[-1] + 1)
				new_y1 = int(start + (row_zero if len(split_ys) != n else row_nonzero)[0])
				new_y2 = int(start + (row_zero if len(split_ys) > 2 else row_nonzero)[-1] + 1)
				if new_y2 - new_y1 >= min_height:
					splitted.append((gid, (new_x1, new_x2, new_y1, new_y2)))
					gid += 1
					split_ys.pop(0)
				else:
					split_ys.pop(1)
			else:
				split_ys.pop(1)

		while len(split_ys) >= 2:
			help()

	return splitted

def process(chapter, page, marginal_text_avg_threshold=10, marginal_text_width_threshold=40):
	st = time.time()

	page_folder = f"./{chapter}/{page}"
	groups_folder = f"{page_folder}/groups"
	
	if os.path.exists(page_folder):
		shutil.rmtree(page_folder)
	os.makedirs(page_folder)

	arr = np.array(Image.open(f"{page_folder}.png").convert("L"))
	black_on_white = arr
	white_on_black_normalized = (255 - arr) / 255

	group_arr = label_groups(white_on_black_normalized)
	raw_boxes = extract_boxes(group_arr)

	def update_group_arr(boxes):
		for gid, (x1, x2, y1, y2) in boxes:
			group_arr[y1:y2, x1:x2][group_arr[y1:y2, x1:x2] != -1] = gid

	boxes = None
	if len(raw_boxes) > 0:
		create_rgb_image(group_arr, colors_rgb, raw_boxes).save(f"{page_folder}/raw_groups.png")

		splitted_boxes = split_on_column(raw_boxes, white_on_black_normalized, group_arr)
		update_group_arr(splitted_boxes)
		create_rgb_image(group_arr, colors_rgb, splitted_boxes).save(f"{page_folder}/raw_splitted_groups.png")

		merged_boxes = sorted(merge_boxes(splitted_boxes), key=lambda b: -b[1][0])
		update_group_arr(merged_boxes)
		create_rgb_image(group_arr, colors_rgb, merged_boxes).save(f"{page_folder}/merged_groups.png")

		cols = split_on_ponctuation(merged_boxes, white_on_black_normalized, group_arr)
		update_group_arr(cols)
		create_rgb_image(group_arr, colors_rgb, cols).save(f"{page_folder}/merged_splitted_groups.png")

		boxes = split_on_chooonpu(cols, white_on_black_normalized, group_arr)
		update_group_arr(boxes)
	else:
		boxes = raw_boxes

	create_rgb_image(group_arr, colors_rgb, boxes).save(f"{page_folder}/groups.png")
	
	# gids_with_marginal_text = []

	# for gid, (x1, x2, y1, y2) in boxes:
	# 	below_avg_count = 0
	# 	if x2 - x1 >= marginal_text_width_threshold:
	# 		avg = col_avg.mean() / 2
	# 		values_after_max = col_avg[np.argmax(col_avg) + 1:]
	# 		below_avg_count = (values_after_max < avg).sum()
	# 		if below_avg_count >= marginal_text_avg_threshold:
	# 			gids_with_marginal_text.append(gid)

	# if len(gids_with_marginal_text) > 0:
	# 	print(gids_with_marginal_text)

	text_img = Image.fromarray(np.ones_like(black_on_white) * 255)

	for gid, (x1, x2, y1, y2) in boxes:
		os.makedirs(f"{groups_folder}/{gid:02}", exist_ok=True)
		crop_img = Image.fromarray(np.pad(np.where(group_arr == gid, black_on_white, 255)[y1:y2, x1:x2], ((15, 15), (15, 15)), mode='constant', constant_values=255))
		crop_img.save(f"{groups_folder}/{gid:02}/{x1}_{x2}_{y1}_{y2}.png")
		text = ocr(crop_img)
		with open(f"{groups_folder}/{gid:02}/text.txt", "w", encoding="utf-8") as f:
			f.write(text)

		def plot_avg(axis, plot_path):
			plt.clf()
			avg = np.mean(np.where(group_arr == gid, white_on_black_normalized, 0)[y1:y2, x1:x2], axis=axis)
			binary = [avg.max() if v > 0 else 0 for v in avg]
			plt.plot(avg)
			plt.plot(binary)
			plt.axhline(y=avg.mean(), color='red')
			plt.savefig(plot_path)
		
		plot_avg(0, f"{groups_folder}/{gid:02}/avg_col_plot.png")
		plot_avg(1, f"{groups_folder}/{gid:02}/avg_row_plot.png")

		draw_col(text_img, text, (x1, y1))
	
	text_img.save(f"{page_folder}/text.png")

	print(chapter, page, time.time() - st)

def main():
	for chapter in range(1, 5+1):
		for page_filename in os.listdir(f"./{chapter}"):
			if not page_filename.endswith(".png"):
				continue
			page = page_filename.split(".")[0]

			if int(page) < 8:
				continue

			process(chapter, page)

# cProfile.run('main()')

main()

# process(1, "008")
# process(1, "101")
