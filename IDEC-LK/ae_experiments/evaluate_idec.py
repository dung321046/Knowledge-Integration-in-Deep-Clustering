import sys

sys.path.append("..")
from experiments.utils_setup import *
import json
import statistics
import numpy as np

INF = 1000000


def element_to_str(e):
    if type(e) == list:
        if len(e) == 1:
            e = e[0]
        elif len(e) == 0:
            return "-"
    if type(e) == int:
        return "{:d}".format(e)
    if type(e) == str:
        return e
    if type(e) == tuple and len(e) == 2:
        if abs(e[0]) < 10 ** -7 and abs(e[1]) < 10 ** -7:
            return "0 $\\pm$ 0".format(e[0], e[1])
        if abs(e[0]) < 0.01:
            return "{:5.2e} $\\pm$ {:5.2e}".format(e[0], e[1])
        if abs(e[1]) < 0.00001:
            return "{:.3f}".format(e[0])
        if e[0] > 100:
            if e[0] > INF:
                return "{:5.2e} $\\pm$ {:5.2e}".format(e[0], e[1])
            return "{:.0f} $\\pm$ {:.0f}".format(e[0], e[1])
        return "{:.4f} $\\pm$ {:.4f}".format(e[0], e[1])
    return "{:.4f}".format(e)


def arr_to_table_latex(arr, column_names, hidden_cols, caption="", ref=""):
    string_file = "\\begin{table}[ht]\n"
    string_file += "\\caption{" + caption + "}\\label{tab:" + ref + "}\n" + r"\resizebox{\columnwidth}!{" + "\n\\begin{tabular}{ |"
    for i in range(len(arr[0])):
        if i in hidden_cols:
            string_file += " H "
        else:
            string_file += " r |"
    string_file += "}\n\\hline\n"
    for i in range(len(arr[0])):
        string_file += column_names[i] + " & "
    string_file = string_file[:-2] + " \\\\  "
    previous_data = ""
    for i in range(len(arr)):
        if previous_data != arr[i][0]:
            string_file = string_file[:-2] + " \\hline \n"
            previous_data = arr[i][0]
        elif i % 2 == 0:
            string_file = string_file[:-2] + " \\hdashline \n"
        for j in range(len(arr[i])):
            string_file += element_to_str(arr[i][j]) + " & "
        string_file = string_file[:-2] + "\\\\ \n"
    string_file += "\\hline\n\\end{tabular}\n}\n"

    string_file += "\\end{table}"
    return string_file


def get_mean_and_std(arr):
    if len(arr) == 0:
        return -1000000, 0.0
    if len(arr) == 1:
        return arr[0], 0.0
    if type(arr[0]) == str:
        return arr[0]
    return statistics.mean(arr), np.std(np.asarray(arr))


if __name__ == "__main__":
    parser = clustering_parser("Evaluate Deep Clustering")
    parser.add_argument('--lr', type=float, default=0.001, metavar='N',
                        help='learning rate for training (default: 0.001)')
    args = parser.parse_args()
    nmi = []
    acc = []
    cluster_loss, recon_loss = [], []
    index = []
    exported_table = []
    for seed in range(5):
        with open("./model/idec_" + args.data + "/" + str(args.lr) + "_" + str(seed) + ".json", "r") as stat_json:
            stat = json.load(stat_json)
        print(stat)
        index.append(seed)
        s = stat[-1]
        nmi.append(s["nmi"])
        acc.append(s["acc"])
        row = [args.data, seed]
        cluster_loss.append(s["cluster-loss"])
        recon_loss.append(s["recon-loss"])
        row.append(nmi[-1])
        row.append(acc[-1])
        row.append(cluster_loss[-1])
        row.append(recon_loss[-1])
        row.append(cluster_loss[-1] + recon_loss[-1])
        exported_table.append(row)
    with open(args.data + "-raw.tex", "w") as file:
        file.write(arr_to_table_latex(exported_table,
                                      ["Data", "Run", "NMI", "ACC", "cluster-loss", "recon-loss", "total-loss"], [],
                                      caption="Raw training results on " + args.data + " with IDEC",
                                      ref="idec-raw-" + args.data))
    exported_table = []
    for data in ["MNIST", "Fashion", "Reuters"]:
        nmi = []
        acc = []
        loss = []
        for seed in range(5):
            with open("./model/idec_" + data + "-sdae/0.001_" + str(seed) + ".json", "r") as stat_json:
                stat = json.load(stat_json)
                s = stat[-1]
                nmi.append(s["nmi"])
                acc.append(s["acc"])
                loss.append(s["cluster-loss"] + s["recon-loss"])
        row = [data, "SDAE+IDEC"]
        row.append(get_mean_and_std(nmi))
        row.append(get_mean_and_std(acc))
        row.append(get_mean_and_std(loss))
        exported_table.append(row)
        print(loss)
    with open("idec-sdae.tex", "w") as file:
        file.write(arr_to_table_latex(exported_table,
                                      ["Data", "Model", "NMI", "ACC", "total-loss"], [4],
                                      caption="SDAE+IDEC performance on MNIST, Fashion and Reuters",
                                      ref="idec-sdae"))
