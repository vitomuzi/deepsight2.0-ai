index = 0
file_path = 'train_all.txt'
file1 = 'train.txt'
file2 = 'valid.txt'

with open(file_path, 'r') as file:
	lines = file.readlines()

with open(file2 , 'w') as fw1:
	for line in lines:
		if index%3 == 2:
			fw1.write(line)
		index += 1
