for dir in */; do
	renamed_folder="$(echo "$dir" | sed -E 's/[^0-9_]+//g' | sed -E 's/_+/_/g')"
	mv "$dir" "$renamed_folder"
	cd "$renamed_folder"
	for file in *; do
		renamed_file="$(echo "$file" | sed -E 's/[^0-9_]+//g' | sed -E 's/_+/_/g' | sed -E 's/(.*)\./\1/')$(echo "$file" | sed -E 's/.*\./\./')"
		mv "$file" "${renamed_file:2}"
		jpegqs "${renamed_file:2}" "${renamed_file:2}"
	done
	cd ..
done
