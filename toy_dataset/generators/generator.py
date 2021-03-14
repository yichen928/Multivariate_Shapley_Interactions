import json
import os
import random
import argparse


parser = argparse.ArgumentParser()
parser.add_argument("--task", help="two choices of toy tasks [multiple_add, and_or]")
parser.add_argument("--min_len", default=5, help="min length of each generated data item")
parser.add_argument("--max_len", default=7, help="max length of each generated data item")
parser.add_argument("--size", default=1000, help="size of dataset")
parser.add_argument("--output_dir", default="../datasets", help="output directory of generated dataset")
args = parser.parse_args()

if args.task == "multiple_add":
    op = ["*", "+"]
else:
    assert args.task == "and_or"
    op = ["&", "|"]

len_range = [args.min_len, args.max_len]
size = args.size
output = os.path.join(args.output_dir, "toy_dataset_%s.json"%args.task)
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
        if ch == op[0]:
            label.append(1)
        else:
            label.append(0)
    return tuple(label)


dataset = []
for _ in range(size):
    length = random.randint(len_range[0], len_range[1])
    ops = []
    for i in range(length-1):
        ops.append(random.choice(op))
    data = generate_data(ops)  # generate data
    label = generate_label(ops)  # generate ground truth

    dataset.append((data, label))

with open(output, "w") as f:
    json.dump(dataset, f, indent=4)