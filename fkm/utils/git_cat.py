import os


def parse1():
	in_dir = '.git/objects'
	out_dir = 'tmp'
	if not os.path.exists(out_dir):
		os.makedirs(out_dir)
	for dir_ in os.listdir(in_dir):
		dir2 = os.path.join(in_dir, dir_)
		if not os.path.isdir(dir2): continue
		for f in os.listdir(dir2):
			name = f'{dir_}{f}'
			out = os.path.join(out_dir, name)
			cmd = f"git cat-file -p {name} > {out}.txt"
			print(cmd)
			os.system(cmd)



def parse():
	in_dir = '.git/lost-found/other'
	out_dir = 'tmp'
	if not os.path.exists(out_dir):
		os.makedirs(out_dir)

	for f in os.listdir(in_dir):
		name = f'{f}'
		out = os.path.join(out_dir, name)
		cmd = f"git cat-file -p {name} > {out}.txt"
		print(cmd)
		os.system(cmd)


parse()


