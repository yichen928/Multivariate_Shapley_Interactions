import json
import numpy as np
import matplotlib.pyplot as plt
import matplotlib
matplotlib.use("agg")

colors = (
    ((220,20,60), (255,182,193), (255,192,203)),  # red
    ((0,0,255), (135,206,250), (135,206,235)),  # blue
    ((0,128,0), (144,238,144), (152,251,152)),    # green
    ((255,165,0), (250,250,210), (255,255,0)),  # yellow
    ((255,0,255), (218,112,214), (221,160,221))   # violet
)
# 线/区域边缘/区域

# models = ["Bert", "ELMo", "LSTM", "CNN", "Transformer"]
models = ["bert", "elmo"]

dataset = "sst-2"


def RGB_to_Hex(rgb):
    strs = "#"
    for i in rgb:
        num = int(i)  # 将str转int
        # 将R、G、B分别转化为16进制拼接转换并大写
        strs += str(hex(num))[-2:].replace('x', '0').upper()

    return strs


x = np.arange(1, 101)

for id, model in enumerate(models):
    if model == "LSTM":
        multi = 100
        continue
    elif model == "Transformer":
        multi = 1000
        continue
    else:
        multi = 1

    with open("difference_%s_%s.json"%(dataset, model.lower()), "r") as f:
        res = json.load(f)
    np_diff = np.zeros((len(res) - 1, 100))
    for i, item in enumerate(res[1:]):
        seg_len = item["seg"][1] - item["seg"][0]
        gt_score = item["gt_score"]
        if "opt_res" in item:
            loss_item = item["opt_res"]
        else:
            loss_item = item["loss_result"]
        for j, each in enumerate(loss_item):
            np_diff[i, j] = abs(each["loss"] - gt_score) * multi
            # np_diff[i, j] = abs(-1 * each["loss"] - gt_score) / (seg_len-1)
    # np_diff = np.random.randn(1000, 100)
    np_std = np_diff.std(0)
    # np_std = np_diff.std(0) / np.sqrt(len(res) - 1)
    np_mean = np_diff.mean(0)

    mean_plus_std = np_mean + np_std
    mean_minus_std = np_mean - np_std
    plt.plot(x, np_mean, color=RGB_to_Hex(colors[id][0]), linewidth=2.5, label=models[id])
    plt.plot(x, mean_plus_std, color=RGB_to_Hex(colors[id][1]), linestyle='--', linewidth=0.5)
    plt.plot(x, mean_minus_std, color=RGB_to_Hex(colors[id][1]), linestyle='--', linewidth=0.5)
    plt.fill_between(x, mean_plus_std, mean_minus_std, color=RGB_to_Hex(colors[id][2]), alpha=0.5)


plt.xlabel("Epoch", fontsize=25)
plt.ylabel("Difference", fontsize=25)
plt.tick_params(labelsize=20)
plt.legend(fontsize=22)
plt.grid(b=True, which='major', axis='y', alpha=0.6, linewidth=1)  # , linestyle='-.'

plt.savefig("./accuracy_evaluation_%s_std.svg"%dataset, bbox_inches='tight')
plt.show()
