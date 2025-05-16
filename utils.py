import numpy as np
from PIL import Image, ImageDraw, ImageFont

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

def draw_rectangle_on_numpy_arr(img_arr, offsets, color=[255, 0, 0]):
	x1, x2, y1, y2 = offsets
	img_arr[y1:y2, x1 - 1] = color
	img_arr[y1:y2, x2 - 1] = color
	img_arr[y1 - 1, x1:x2] = color
	img_arr[y2 - 1, x1:x2] = color

font_cache = {}

def get_font(font_path, font_size):
	cache_key = (font_path, font_size)
	if cache_key not in font_cache:
		if font_path:
			try:
				font_cache[cache_key] = ImageFont.truetype(font_path, font_size)
			except IOError:
				font_cache[cache_key] = ImageFont.load_default(font_size)
		else:
			font_cache[cache_key] = ImageFont.load_default(font_size)
	return font_cache[cache_key]

def draw_gid(img_arr, number, position, font_size=30):
	number_image = Image.new('L', (font_size * 2, font_size * 2), 255)
	draw = ImageDraw.Draw(number_image)
	font = get_font(None, font_size)
	draw.text((0, 0), str(number).zfill(2), font=font, fill=0)
	number_arr = np.array(number_image)
	x_offset, y_offset = position
	y_offset -= font_size + 1
	mask = number_arr != 255
	number_arr_rgb = np.dstack([number_arr] * 3)
	boundaries = (slice(y_offset, y_offset + number_arr_rgb.shape[0]), slice(x_offset, x_offset + number_arr_rgb.shape[1]))
	img_arr[boundaries] = np.where(mask[..., None], number_arr_rgb, img_arr[boundaries])

def draw_col(img_arr, text, position, font_size=38):
	draw = ImageDraw.Draw(img_arr)
	font = get_font("./fonts/msgothic.ttc", font_size)
	
	x_offset, y_offset = position
	for char in text:
		if char == "ー":
			char = "|"
		draw.text((x_offset, y_offset), char, font=font, fill=0)
		y_offset += font_size

def create_rgb_image(group_arr, colors_rgb, boxes):
	rgb_arr = np.ones((*group_arr.shape, 3), dtype=np.uint8) * 255
	for gid, (x1, x2, y1, y2) in boxes:
		rgb_arr[y1:y2, x1:x2][group_arr[y1:y2, x1:x2] == gid] = colors_rgb[gid % len(colors_rgb)]
		draw_rectangle_on_numpy_arr(rgb_arr, (x1, x2, y1, y2))
		try:
			draw_gid(rgb_arr, gid, (x1, y1))
		except:
			# too lazy to fix
			# img_arr[boundaries] = np.where(mask[..., None], number_arr_rgb, img_arr[boundaries])
			# ValueError: operands could not be broadcast together with shapes (60,60,1) (60,60,3) (0,60,3)
			pass
	return Image.fromarray(rgb_arr)