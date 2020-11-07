import os


def count_line(filepath):
    count = 0
    f = open(filepath, "r")
    for line in f.readlines():
        count = count + 1
    return count

def find_ckpt(ckpt_path):
    files = os.listdir(ckpt_path)
    for fi in files:
        fi_d = os.path.join(ckpt_path, fi)
        if fi.endswith(".ckpt"):
          return fi_d

