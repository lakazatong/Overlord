import time, math, os, shutil, numpy as np
from PIL import Image
from bisect import insort
import matplotlib.pyplot as plt

plt.figure(figsize=(12, 4))

# import cProfile
from manga_ocr import MangaOcr

# image_path = "./1/097.png"
image_path = "./1/111.png"
# image_path = "./1/117.png"
# image_path = "./1/128.png"

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

def circle_offsets(radius):
	offsets = []
	r2 = radius * radius
	for dx in range(-radius, radius + 1):
		for dy in range(-radius, radius + 1):
			d2 = dx*dx + dy*dy
			if d2 <= r2:
				insort(offsets, ((-d2), (dx, dy)))
	return [xy for _, xy in offsets]

circle_offsets_cache = {}

def get_circle_offsets(radius):
	if radius not in circle_offsets_cache:
		circle_offsets_cache[radius] = circle_offsets(radius)
	return circle_offsets_cache[radius]

def generate_coordinates_within_distance(coord, radius, shape):
	x, y = coord
	offsets = get_circle_offsets(radius)
	return [
		(x + dx, y + dy)
		for dx, dy in offsets
		if 0 <= x + dx < shape[0] and 0 <= y + dy < shape[1]
	]

# radius = 100
# shape = (radius*2+1, radius*2+1, 3)
# outer = math.floor(5*radius*math.sqrt(2))
# cur_outer = 0
# arr = np.zeros(shape, dtype=np.uint8)
# for x, y, d in generate_coordinates_within_distance((radius, radius), radius, shape):
# 	if cur_outer < outer:
# 		arr[x, y] = (255, 255, 255)
# 	else:
# 		arr[x, y] = (128, 128, 128)
# 	cur_outer += 1
# Image.fromarray(arr).save("circle.png")

# exit(0)

def label_groups(binary_image, radius=20):
	shape = binary_image.shape
	group_arr = np.full_like(binary_image, -1, dtype=int)
	group_arr[binary_image == 1] = -2
	group_id = 0
	outer = 4*math.floor(radius*math.sqrt(2))
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
			for nx, ny in apply_mask(cx, cy):
				group_arr[nx, ny] = group_id
				queue.append((nx, ny))

	while np.any(group_arr == -2):
		# print(group_id)
		indices = np.transpose(np.where(group_arr == -2))
		x, y = indices[np.random.choice(len(indices))]
		help(x, y)
		group_id += 1
	
	return group_arr

def extract_bounding_boxes(group_arr, spacing_threshold=5, min_width=23):
	boxes = {}
	max_gid = 0
	for y in range(group_arr.shape[0]):
		for x in range(group_arr.shape[1]):
			gid = group_arr[y, x]
			if gid > max_gid:
				max_gid = gid
			if gid != -1:
				if gid not in boxes:
					boxes[gid] = [x, x + 1, y, y + 1]
				else:
					boxes[gid][0] = min(boxes[gid][0], x)
					boxes[gid][1] = max(boxes[gid][1], x + 1)
					boxes[gid][2] = min(boxes[gid][2], y)
					boxes[gid][3] = max(boxes[gid][3], y + 1)

	group_mask = np.where(group_arr != -1, 1, 0)
	col_averages = np.mean(group_mask, axis=0)

	split_boxes = []
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
			split_boxes.append((gid, (x1, x2, y1, y2)))
			continue

		plt.clf()
		plt.plot(col_avg)
		plt.savefig(f"groups/{max_gid + 1:02}_plot.png")

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

			new_x1 = start + col_nonzero[0]
			new_x2 = start + col_nonzero[-1] + 1
			new_y1 = y1 + row_nonzero[0]
			new_y2 = y1 + row_nonzero[-1] + 1

			max_gid += 1
			split_boxes.append((max_gid, (new_x1, new_x2, new_y1, new_y2)))
			i += 1

	return sorted(split_boxes, key=lambda x: x[0])

def draw_rectangle_on_numpy_arr(img_arr, offsets, color=[255, 0, 0]):
	x1, x2, y1, y2 = offsets
	img_arr[y1:y2, x1] = color
	img_arr[y1:y2, x2 - 1] = color
	img_arr[y1, x1:x2] = color
	img_arr[y2 - 1, x1:x2] = color

def create_rgb_image(group_arr, colors_rgb, boxes):
	rgb_arr = np.ones((group_arr.shape[0], group_arr.shape[1], 3), dtype=np.uint8) * 255
	for i in range(group_arr.shape[0]):
		for j in range(group_arr.shape[1]):
			group_id = group_arr[i, j]
			if group_id != -1:
				rgb_arr[i, j] = colors_rgb[group_id % len(colors_rgb)]
	for _, offsets in boxes:
		draw_rectangle_on_numpy_arr(rgb_arr, offsets)
	return rgb_arr

def main():
	if os.path.exists("groups"):
		shutil.rmtree("groups")
	os.makedirs("groups")

	arr = np.array(Image.open(image_path).convert("L"))
	binary_image = (arr != 255).astype(np.uint8)
	group_arr = label_groups(binary_image)
	boxes = extract_bounding_boxes(group_arr)

	for gid, (x1, x2, y1, y2) in boxes:
		print(gid, x2 - x1)

	# return

	Image.fromarray(create_rgb_image(group_arr, colors_rgb, boxes)).save("groups.png")
	ocr = MangaOcr()

	for i, (gid, (x1, x2, y1, y2)) in enumerate(boxes):
		crop_img = Image.fromarray(np.pad(arr[y1:y2, x1:x2], ((15, 15), (15, 15)), mode='constant', constant_values=255))
		crop_img.save(f"groups/{gid:02}.png")
		text = ocr(crop_img)
		with open(f"groups/{gid:02}.txt", "w", encoding="utf-8") as f:
			f.write(text)

# cProfile.run('main()')
main()
