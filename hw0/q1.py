import sys


# Comlumn index and input file
col_idx = int(sys.argv[1])
in_file = sys.argv[2]

# Parse input to a 2D matrix.
with open(in_file) as file:
    mat = [[float(digit) for digit in line.split()] for line in file]

# Extract the selected column and sort it.
col = [row[col_idx] for row in mat]
tmp = sorted(col)
result = ','.join('{1}'.format(*k) for k in enumerate(tmp))

# Write to output file.
out_file = open('ans1.txt', 'w')
out_file.write(result)
out_file.close()

