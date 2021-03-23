import numpy as np
import matplotlib.pyplot as plt
import os
import matplotlib
matplotlib.use("Agg")
# from result.draw_figure import *


def RGB_to_Hex(rgb):
    strs = "#"
    for i in rgb:
        num = int(i)  # 将str转int
        # 将R、G、B分别转化为16进制拼接转换并大写
        strs += str(hex(num))[-2:].replace('x', '0').upper()

    return strs


def draw_figure(pth, model_name, dataset_name, sample_num=5):
    for root, dirs, files in os.walk(pth):
        results = np.array([0 for _ in range(sample_num)])
        print(files)
        count = 0
        for file in files:
            file_path = os.path.join(pth, file)
            result = np.load(file_path)
            results = results + result
            count += 1
        results = results / count
        x = np.array([i for i in range(200, 1001, 200)])
        plt.plot(x, results)
        plt.title("%s_%s" % (model_name, dataset_name))
        plt.savefig(pth + "/curve.png")


def get_instability(result):
    ans_insta = [0 for _ in range(len(result[-1]["result"]))]
    ans_maximal = [0 for _ in range(len(result[-1]["result"]))]
    ans_minimal = [0 for _ in range(len(result[-1]["result"]))]
    x = [0 for _ in range(len(result[-1]["result"]))]
    for i in range(1, len(result)):
        sentence_result = result[i]["result"]

        total_inter = []
        for j in range(len(sentence_result)):
            for k in range(len(sentence_result[j]["True"])):
                for l in range(len(sentence_result[j]["False"])):
                    total_inter.append(0 - sentence_result[j]["True"][k] - sentence_result[j]["False"][l])

        avg_inter = sum(total_inter) / len(total_inter)
        for j in range(len(sentence_result)):
            inter = []
            insta = []
            maximal = []
            minimal = []
            x[j] = sentence_result[j]["m_cnt"] * sentence_result[j]["g_sample_num"]
            for k in range(len(sentence_result[j]["True"])):
                for l in range(len(sentence_result[j]["False"])):
                    inter.append(0 - sentence_result[j]["True"][k] - sentence_result[j]["False"][l])

            for k in range(len(inter)):
                for l in range(k + 1, len(inter)):
                    insta.append(abs(inter[k] - inter[l]) / (abs(avg_inter)))
                    # insta.append(abs(inter[k] - inter[l]) / (0.5 * (abs(inter[k]) + abs(inter[l]))))

            for k in range(len(sentence_result[j]["True"])):
                for l in range(k + 1, len(sentence_result[j]["True"])):
                    tmp1 = sentence_result[j]["True"][k]
                    tmp2 = sentence_result[j]["True"][l]
                    maximal.append(abs(tmp1 - tmp2) / (0.5 * (abs(tmp1) + abs(tmp2))))
            for k in range(len(sentence_result[j]["False"])):
                for l in range(k + 1, len(sentence_result[j]["False"])):
                    tmp1 = sentence_result[j]["False"][k]
                    tmp2 = sentence_result[j]["False"][l]
                    minimal.append(abs(tmp1 - tmp2) / (0.5 * (abs(tmp1) + abs(tmp2))))

            avg_insta = sum(insta) / len(insta)
            avg_maximal = sum(maximal) / len(maximal)
            avg_minimal = sum(minimal) / len(minimal)
            ans_insta[j] += avg_insta
            ans_maximal[j] += avg_maximal
            ans_minimal[j] += avg_minimal
    ans_insta = np.array(ans_insta) / (len(result) - 1)
    ans_maximal = np.array(ans_maximal) / (len(result) - 1)
    ans_minimal = np.array(ans_minimal) / (len(result) - 1)
    x = np.array(x)
    return ans_insta, x


def main(time, model_name, dataset_name):
    instablities = []
    for i in range(len(time)):
        pth = "./result/instability/%s/%s/%s" % (time[i], model_name[i], dataset_name)
        # pth = "./result/instability/%s/%s" % (model_name, dataset_name)
        if not os.path.exists(pth):
            raise RuntimeError("No such directory: %s" % pth)
        # draw_figure(pth, model_name, dataset_name)

        result = np.load(pth + "/result_%s_%s.npy" % (dataset_name, model_name[i]), allow_pickle=True)

        instability, x = get_instability(result)
        instablities.append(instability)
    return instablities, x


if __name__ == '__main__':
    time = ["2020-12-10", "2020-12-10"]
    model_name = ["elmo", "cnn"]
    dataset_name = "sst-2"
    instablility, x = main(time, model_name, dataset_name)

    root = os.getcwd()
    figure_path = root + "/result/instability/%s_curves_2020-12-10.svg" % dataset_name

    colors = (
        # ((220, 20, 60), (255, 182, 193), (255, 192, 203)),  # red
        ((0, 0, 255), (135, 206, 250), (135, 206, 235)),  # blue
        # ((0, 128, 0), (144, 238, 144), (152, 251, 152)),  # green
        ((255, 165, 0), (250, 250, 210), (255, 255, 0)),  # yellow
        # ((255, 0, 255), (218, 112, 214), (221, 160, 221))  # violet
    )

    for id, model in enumerate(model_name):
        plt.plot(x, instablility[id], color=RGB_to_Hex(colors[id][0]), linewidth=2.5)

    plt.xlabel("Epoch", fontsize=25)
    plt.ylabel("Instability", fontsize=25)
    # plt.ylim((0.0, 0.150))
    plt.tick_params(labelsize=20)
    # plt.legend(fontsize=22)
    plt.grid(b=True, which='major', axis='y', alpha=0.6, linewidth=1)  # , linestyle='-.'

    # plt.savefig("./difference_%s_normal.svg"%dataset, bbox_inches='tight')
    plt.savefig(figure_path, dpi=4000, format="svg", bbox_inches='tight')
    plt.show()


