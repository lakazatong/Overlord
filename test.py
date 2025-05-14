import re
from pathlib import Path
import shutil

pattern = re.compile(r'(\d+)_(\d+)_plot\.png')

def find_significant_plots(folder):
	dest = Path('test')
	dest.mkdir(exist_ok=True)
	for path in Path(folder).glob('*/groups/*.png'):
		match = pattern.fullmatch(path.name)
		if match and int(match.group(2)) >= 10:
			image_path = path.parent / f"{int(match.group(1)):02}.png"
			if image_path.exists():
				new_name = f"{path.parent.parent.name}_{match.group(1)}_{match.group(2)}.png"
				shutil.copy(image_path, dest / new_name)

find_significant_plots("1")
