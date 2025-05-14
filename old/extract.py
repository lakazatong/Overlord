import os, math, cv2
from PIL import Image, ImageDraw
import numpy as np
import matplotlib.pyplot as plt
from collections import defaultdict
from bisect import insort

reference_images = []
# elements = []
for ref_path in os.listdir("references"):
	ref = Image.open(f"./references/{ref_path}").convert("L")
	ref.name = ref_path
	reference_images.append(ref)
	# insort(elements, (ref_path, np.mean(np.array(ref))), key=lambda e: e[1])

# for e in elements:
# 	print(e)

# exit(0)

counter = 0

def crop_to_match(arr1, arr2, i, threshold=3):
	def crop_on_threshold(arr):
		mask = arr < 128
		rows = np.any(mask, axis=1)
		cols = np.any(mask, axis=0)
		
		top = np.argmax(rows)
		bottom = len(rows) - np.argmax(np.flip(rows))
		left = np.argmax(cols)
		right = len(cols) - np.argmax(np.flip(cols))
		
		return arr[top:bottom, left:right]

	arr1 = crop_on_threshold(arr1)
	arr2 = crop_on_threshold(arr2)

	h1, w1 = arr1.shape[:2]
	h2, w2 = arr2.shape[:2]

	if abs(h1 - h2) > threshold or abs(w1 - w2) > threshold:
		return None, None

	# if counter == 7 and i == 40:
	# 	print(h1, w1)
	# 	print(h2, w2)

	target_w = min(w1, w2)
	target_h = min(h1, h2)

	if w1 > target_w:
		arr1 = arr1[:, :target_w]
	if w2 > target_w:
		arr2 = arr2[:, :target_w]

	if h1 > target_h:
		diff = h1 - target_h
		top = diff // 2
		arr1 = arr1[top:top + target_h]
	if h2 > target_h:
		diff = h2 - target_h
		top = diff // 2
		arr2 = arr2[top:top + target_h]

	return arr1, arr2

def find_closest_match(target, i=0):
	best_score = float('inf')
	best_match = None
	target_height, target_width = target.shape[:2]
	# data = []

	for ref in reference_images:
		ref_width, ref_height = ref.size

		# data.append((ref_height, target_height, abs(ref_height - target_height)))
		if abs(ref_height - target_height) > 9:
			continue

		cropped_ref, cropped_target = crop_to_match(np.array(ref), target, i)
		if cropped_ref is None or cropped_target is None:
			continue

		score = np.sum((cropped_ref - cropped_target) ** 2) / float(cropped_ref.shape[0] * cropped_ref.shape[1])

		# if counter == 7 and i == 40:
		# 	print(ref.name, score)
		# 	if ref.name == "double_exclamation.png":
		# 		Image.fromarray(cropped_ref).save("./cropped_ref.png")
		# 		Image.fromarray(cropped_target).save("./cropped_target.png")
		
		if score < best_score:
			best_score = score
			best_match = ref

	# if best_match is None:
	# 	print()
	# 	Image.fromarray(target).save("./target.png")
	# 	for d in data:
	# 		print(d)

	return best_match, best_score

# match_img, match_score = find_closest_match(np.array(Image.open("./target.png").convert("L")))
# print(match_img.name, match_score)

# exit(0)

def find_col_gaps(col_avg, min_width=28, max_width=37):
	gaps = []
	start = None
	i = 0
	for tmp, avg in enumerate(col_avg):
		print(tmp, avg)
	while i < len(col_avg):
		if col_avg[i] < 255:
			if start is None:
				start = i
		elif not (start is None):
			end = min(i, start + max_width)
			if end - start >= min_width:
				gaps.append((start, end))
			start = None
		i += 1

	if not (start is None):
		end = min(len(col_avg), start + max_width)
		if end - start >= min_width:
			gaps.append((start, end))

	return gaps

def find_row_gaps(col):
	print()
	global counter
	row_avg = np.mean(col, axis=1)
	gaps = []
	in_gap = False
	start = None
	
	for i, val in enumerate(row_avg):
		if val < 255:
			if not in_gap:
				in_gap = True
				start = i
				# if counter == 10:
				# 	print("start", start)
		elif in_gap:
			n = i - start
			if n > 1 and sum(row_avg[start:i]) / n < 250:
				gaps.append((start, i))
			in_gap = False
	
	if in_gap:
		gaps.append((start, len(row_avg)))
	
	for i, (start, end) in enumerate(gaps):
		match_img, match_score = find_closest_match(col[start:end], i)
		if match_img is None:
			gaps[i] = (start, end, False)
			continue
		print(counter, i, (start, end), match_img.name if match_img else None, match_score)
		gaps[i] = (start, end, True)
		# if counter == 7 and i == 40:
		# 	print()
		# 	print(counter, i, match_img.name, match_score)
		# 	Image.fromarray(col[start:end]).save("./target.png")
		# 	match_img.save("./match_img.png")

	counter += 1

	return gaps

def find_grid(image_path):
	arr = np.array(Image.open(image_path))
	arr = arr[:-200]

	col_avg = np.mean(arr, axis=0)
	col_gaps = find_col_gaps(col_avg)
	
	rows_gaps = []
	for start, end in col_gaps:
		rows_gaps.append(find_row_gaps(arr[:, start:end]))

	return col_gaps, rows_gaps

def visualize_gaps(image_path, col_gaps, rows_gaps, output_path=None):
	arr = np.array(Image.open(image_path).convert('RGB'))
	for i, row_gaps in enumerate(rows_gaps):
		col_gap_start, col_gap_end = col_gaps[i]
		arr[:, col_gap_start:col_gap_end, 0] = 128
		for row_gap_start, row_gap_end, is_special in row_gaps:
			arr[row_gap_start:row_gap_end, col_gap_start:col_gap_end, 2] += 64 if is_special else 196

	result_img = Image.fromarray(arr)
	result_img.save(output_path or image_path)

img_path = "./1/111.png"

col_gaps, rows_gaps = find_grid(img_path)

print(len(col_gaps), max(len(row_gap) for row_gap in rows_gaps))

visualize_gaps(img_path, col_gaps, rows_gaps, "./gaps.png")