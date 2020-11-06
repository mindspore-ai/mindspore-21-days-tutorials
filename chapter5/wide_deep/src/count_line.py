import argparse


def count_line(filepath):
    count = 0
    f = open(filepath, "r")
    for line in f.readlines():
        count = count + 1
    return count


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Count line nums of dataset")
    parser.add_argument("--data_path", type=str, default="/wide_deep/data/one_percent/",
                        help='The path of the data file')
    parser.add_argument("--file_name", type=str, default="mini_demo.txt",
                        help='The name of the data file')
    args = parser.parse_args()
    data_file = args.data_path + args.file_name
    line_num = count_line(data_file)
    print("{} line num: {}".format(args.file_name, line_num))

