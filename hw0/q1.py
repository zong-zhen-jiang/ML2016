import sys


# Comlumn index and input file
col = int(sys.argv[1])
in_file = sys.argv[2]

# Parse input to a 2D matrix.
with open(in_file) as file:
    mat = [[float(digit) for digit in line.split()] for line in file]

# Transpose the matrix.
mat_t = zip(*mat)

# Sort the selected column.
tmp = sorted(mat_t[col])
result = ','.join('{1}'.format(*k) for k in enumerate(tmp))

# Write to output file.
out_file = open('ans1.txt', 'w')
out_file.write(result)
out_file.close()

