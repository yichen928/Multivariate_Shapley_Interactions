import json
import numpy as np
import matplotlib.pyplot as plt
import matplotlib
matplotlib.use("agg")

layers = (0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11)

if __name__ == "__main__":
    with open("interaction_sst-2_bert_layer.json", "r") as f:
        res_sst = json.load(f)
    with open("interaction_cola_bert_layer.json", "r") as f:
        res_cola = json.load(f)

    inter_sst = np.zeros((len(res_sst)-1, len(res_sst[1])-3))
    for i in range(1, len(res_sst)):
        for j, layer in enumerate(layers):
            inter_sst[i-1, j] = res_sst[i][str(j)]["opt_res_max"][-1]["loss"]*-1 - res_sst[i][str(j)]["opt_res_min"][-1]["loss"]

    inter_cola = np.zeros((len(res_cola)-1, len(res_cola[1])-3))
    for i in range(1, len(res_cola)):
        for j, layer in enumerate(layers):
            # inter_cola[i-1, j] = res_cola[i][str(layer)]["gt"]["max_score"]-res_cola[i][str(layer)]["gt"]["min_score"]
            inter_cola[i-1, j] = res_cola[i][str(j)]["opt_res_max"][-1]["loss"]*-1 - res_cola[i][str(j)]["opt_res_min"][-1]["loss"]

    inter_sst = np.mean(inter_sst, axis=0)
    inter_cola = np.mean(inter_cola, axis=0)

    x = list(range(1, len(layers)+1))

    plt.plot(x, inter_sst, label="SST-2", linewidth=2.5)
    plt.plot(x, inter_cola, label="CoLA", linewidth=2.5)
    plt.legend(fontsize=22)
    plt.xlabel("Layer", fontsize=25)
    plt.ylabel("Interaction", fontsize=25)

    plt.tick_params(labelsize=20)
    plt.xticks(np.arange(0, 13, 1))
    plt.yticks(np.arange(0, 1.6, 0.3))

    # plt.title("Interaction by Layer Bert")
    plt.grid(b=True, which='major', axis='y', alpha=0.6, linewidth=1)  # , linestyle='-.'

    plt.savefig('interaction_bert_layer.svg', bbox_inches='tight')
    plt.show()
