import cProfile, time, math, os, shutil, numpy as np
from PIL import Image, ImageDraw, ImageFont
from collections import defaultdict
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

def label_groups(arr, radius=15):
	shape = arr.shape
	group_arr = np.full_like(arr, -1, dtype=int)
	group_arr[arr != 255] = -2
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
		# print(group_id)
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
	
	return [(gid, box) for gid, box in sorted(boxes.items()) if (box[1] - box[0]) * (box[3] - box[2]) > min_area]

def split_boxes(boxes, group_arr, spacing_threshold=5, min_width=23):
	splitted = []
	max_gid = max(boxes, key=lambda x: x[0])[0]
	for gid, (x1, x2, y1, y2) in boxes:
		local_group_mask = np.where(group_arr == gid, 1, 0)
		col_avg = np.mean(local_group_mask[y1:y2, x1:x2], axis=0)

		binary = [1 if v > 0 else 0 for v in col_avg]

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

		if len(zero_ranges) == 0:
			splitted.append((gid, (x1, x2, y1, y2)))
			continue
		
		mids = [x1 + (start + end) // 2 for start, end in zero_ranges]
		split_xs = [x1] + mids + [x2]
		nb_added = 0
		i = 0
		while i < len(split_xs) - 1:
			start = split_xs[i]
			end = split_xs[i + 1]
			width = end - start
			if width < min_width:
				split_xs.pop(i + 1)
				continue

			col_nonzero = np.nonzero(np.mean(local_group_mask[y1:y2, start:end], axis=0))[0]
			row_nonzero = np.nonzero(np.mean(local_group_mask[y1:y2, start:end], axis=1))[0]

			if col_nonzero.size == 0:
				print("impossible case reached")
				print(y1, y2)
				print(start, end)
				print(np.mean(local_group_mask[y1:y2, start:end]))
				print(np.mean(local_group_mask[y1:y2, start:end]))
				exit(1)

			if row_nonzero.size == 0:
				print("impossible case reached")
				print(y1, y2)
				print(start, end)
				print(np.mean(local_group_mask[y1:y2, start:end]))
				print(np.mean(local_group_mask[y1:y2, start:end]))
				exit(1)

			new_x1 = int(start + col_nonzero[0])
			new_x2 = int(start + col_nonzero[-1] + 1)
			new_y1 = int(y1 + row_nonzero[0])
			new_y2 = int(y1 + row_nonzero[-1] + 1)

			max_gid += 1
			splitted.append((max_gid, (new_x1, new_x2, new_y1, new_y2)))
			nb_added += 1
			i += 1

		if nb_added > 1:
			print(mids)

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

def draw_rectangle_on_numpy_arr(img_arr, offsets, color=[255, 0, 0]):
	x1, x2, y1, y2 = offsets
	img_arr[y1:y2, x1 - 1] = color
	img_arr[y1:y2, x2] = color
	img_arr[y1 - 1, x1:x2] = color
	img_arr[y2, x1:x2] = color

def draw_number_on_image(img_arr, number, position, font_size=30):
	number_image = Image.new('L', (font_size * 2, font_size * 2), 255)
	draw = ImageDraw.Draw(number_image)
	try:
		font = ImageFont.truetype("arial.ttf", font_size)
	except IOError:
		font = ImageFont.load_default(font_size)
	draw.text((0, 0), str(number).zfill(2), font=font, fill=0)
	number_arr = np.array(number_image)
	
	x_offset, y_offset = position
	y_offset -= font_size + 1

	mask = number_arr != 255
	number_arr_rgb = np.dstack([number_arr] * 3)
	
	boundaries = (slice(y_offset, y_offset + number_arr_rgb.shape[0]), slice(x_offset, x_offset + number_arr_rgb.shape[1]))
	img_arr[boundaries] = np.where(mask[..., None], number_arr_rgb, img_arr[boundaries])

def create_rgb_image(group_arr, colors_rgb, boxes):
	rgb_arr = np.ones((*group_arr.shape, 3), dtype=np.uint8) * 255
	for gid, (x1, x2, y1, y2) in boxes:
		rgb_arr[y1:y2, x1:x2][group_arr[y1:y2, x1:x2] == gid] = colors_rgb[gid % len(colors_rgb)]
		draw_rectangle_on_numpy_arr(rgb_arr, (x1, x2, y1, y2))
		try:
			draw_number_on_image(rgb_arr, gid, (x1, y1))
		except:
			# too lazy to fix
			# img_arr[boundaries] = np.where(mask[..., None], number_arr_rgb, img_arr[boundaries])
			# ValueError: operands could not be broadcast together with shapes (60,60,1) (60,60,3) (0,60,3)
			pass
	return Image.fromarray(rgb_arr)

def process(chapter, page, marginal_text_avg_threshold=10, marginal_text_width_threshold=40):
	print()
	st = time.time()

	page_folder = f"./{chapter}/{page}"
	groups_folder = f"{page_folder}/groups"
	
	if os.path.exists(page_folder):
		shutil.rmtree(page_folder)
	os.makedirs(groups_folder)

	arr = np.array(Image.open(f"{page_folder}.png").convert("L"))

	group_arr = label_groups(arr)
	raw_boxes = extract_boxes(group_arr)

	create_rgb_image(group_arr, colors_rgb, raw_boxes).save(f"{page_folder}/raw_groups.png")
	
	boxes = []
	if len(raw_boxes) > 0:
		splitted_boxes = split_boxes(raw_boxes, group_arr)

		for gid, (x1, x2, y1, y2) in splitted_boxes:
			group_arr[y1:y2, x1:x2][group_arr[y1:y2, x1:x2] != -1] = gid
		create_rgb_image(group_arr, colors_rgb, splitted_boxes).save(f"{page_folder}/splitted_groups.png")

		boxes = sorted(merge_boxes(splitted_boxes), key=lambda b: -b[0])
		for gid, (x1, x2, y1, y2) in boxes:
			group_arr[y1:y2, x1:x2][group_arr[y1:y2, x1:x2] != -1] = gid

	gids_with_marginal_text = []

	for gid, (x1, x2, y1, y2) in boxes:
		plt.clf()
		col_avg = np.mean(np.where(group_arr == gid, 1, 0)[y1:y2, x1:x2], axis=0)
		binary = [col_avg.max() if v > 0 else 0 for v in col_avg]
		plt.plot(col_avg)
		plt.plot(binary)
		plt.axhline(y=col_avg.mean(), color='red')

		below_avg_count = 0
		if x2 - x1 >= marginal_text_width_threshold:
			avg = col_avg.mean() / 2
			values_after_max = col_avg[np.argmax(col_avg) + 1:]
			below_avg_count = (values_after_max < avg).sum()
			if below_avg_count >= marginal_text_avg_threshold:
				gids_with_marginal_text.append(gid)

		plt.savefig(f"{groups_folder}/{gid:02}_{below_avg_count}_plot.png")

	if len(gids_with_marginal_text) > 0:
		print(gids_with_marginal_text)

	create_rgb_image(group_arr, colors_rgb, boxes).save(f"{page_folder}/groups.png")

	for gid, (x1, x2, y1, y2) in boxes:
		crop_img = Image.fromarray(np.pad(arr[y1:y2, x1:x2], ((15, 15), (15, 15)), mode='constant', constant_values=255))
		crop_img.save(f"{groups_folder}/{gid:02}.png")
		text = ocr(crop_img)
		with open(f"{groups_folder}/{gid:02}.txt", "w", encoding="utf-8") as f:
			f.write(text)
	
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

# process(1, "101")
