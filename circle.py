def generate_circle_ascii(n):
	size = 2 * n + 1
	result = []
	for y in range(n, -n - 1, -1):
		row = []
		for x in range(-n, n + 1):
			r2 = x * x + y * y
			if r2 > n * n:
				row.append(".")
			else:
				is_boundary = False
				for dx, dy in [(1,0), (-1,0), (0,1), (0,-1)]:
					nx, ny = x + dx, y + dy
					if nx * nx + ny * ny > n * n:
						is_boundary = True
						break
				row.append("x" if is_boundary else "o")
		result.append(" ".join(row))
	result[0] = "\t" + result[0]
	return "\n\t".join(result)

if __name__ == "__main__":
	for n in range(11):
		print(f"{n = }")
		print(generate_circle_ascii(n))
