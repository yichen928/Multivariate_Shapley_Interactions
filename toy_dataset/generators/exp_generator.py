import json
import os
import random
import argparse

parser = argparse.ArgumentParser()
parser.add_argument("--min_len", default=5, help="min length of each generated data item")
parser.add_argument("--max_len", default=7, help="max length of each generated data item")
parser.add_argument("--size", default=1000, help="size of dataset")
parser.add_argument("--output_dir", default="../datasets", help="output directory of generated dataset")
args = parser.parse_args()

op = ["**", "*", "+"]
len_range = [args.min_len, args.max_len]
size = args.size
output = os.path.join(args.output_dir, "toy_dataset_exp.json")
array_name = "a"


def generate_data(ops):
    s = ""
    for i in range(len(ops)):
        s += "%s[%d]"%(array_name, i)
        s += ops[i]
    s += "%s[%d]"%(array_name, len(ops))
    return s


def generate_label(ops):
    label = []
    for ch in ops:
        if ch == op[2]:
            label.append(0)
        else:
            label.append(1)
    return tuple(label)


dataset = []
for _ in range(size):
    length = random.randint(len_range[0], len_range[1])
    ops = []
    for i in range(length-1):
        if i == 0 or ops[-1] == "+":
            ops.append(random.choice(op))
        elif ops[-1] == "*":
            ops.append(random.choice(op[1:]))
        else:
            ops.append("+")
    data = generate_data(ops)
    label = generate_label(ops)

    dataset.append((data, label))

with open(output, "w") as f:
    json.dump(dataset, f, indent=4)