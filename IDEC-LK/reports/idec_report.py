import statistics
import json
import numpy as np

INF = 1000000

import os


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


def write_stat(filename, caption, data, models, ref="tb-TBD", hidden_cols=[]):
    exported_table = []
    for i in range(len(data)):
        for fname, name in models.items():
            row = [data[i], name]
            nmi = []
            acc = []
            cluster_loss, recon_loss = [], []
            index = []
            for seed in range(5):
                full_name = "../ae_experiments/model/idec_" + data[i] + "/" + fname + "_" + str(seed) + ".json"
                if not os.path.isfile(full_name):
                    continue
                with open(full_name, "r") as stat_json:
                    stat = json.load(stat_json)
                print(stat)
                index.append(seed)
                s = stat[-1]
                nmi.append(s["nmi"])
                acc.append(s["acc"])
                cluster_loss.append(s["cluster-loss"])
                recon_loss.append(s["recon-loss"])
            row.append(get_mean_and_std(nmi))
            row.append(get_mean_and_std(acc))
            row.append(get_mean_and_std(cluster_loss))
            row.append(get_mean_and_std(recon_loss))
            exported_table.append(row)
    print(exported_table)
    with open(filename + ".tex", "w") as file:
        file.write(arr_to_table_latex(exported_table,
                                      ["Data", "Models", "NMI", "ACC", "cluster-loss", "recon-loss"], hidden_cols,
                                      caption=caption,
                                      ref=ref))


if __name__ == '__main__':
    data = ["MNIST", "Fashion", "Reuters"]
    models = {"0.001": "lr=0.001"}
    write_stat("./idec", "IDEC performance without pretrain", data, models, ref="pw", hidden_cols=[7])
