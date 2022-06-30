import json
import statistics

import matplotlib.pyplot as plt
import numpy as np
from scipy.stats import ks_2samp

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
        if e[0] > 1:
            return "{:.2f} $\\pm$ {:.2f}".format(e[0], e[1])
        return "{:.4f} $\\pm$ {:.4f}".format(e[0], e[1])
    return "{:.4f}".format(e)


def element_to_str2(e, b):
    ans = element_to_str(e)
    if b == "best":
        ans = "{ \\color{green} " + ans + "}"
    elif b == "sim":
        ans = "{ \\color{blue} " + ans + "}"
    return ans


def arr_to_table_latex(arr, arr_best, column_names, hidden_cols, caption="", ref=""):
    string_file = "\\begin{table}[ht]\n"
    string_file += "\\caption{" + caption + "}\\label{tab:" + ref + "}\n" + r"\resizebox{\columnwidth}!{" + "\n\\begin{tabular}{ |"
    ncolumn = len(column_names)
    for i in range(ncolumn):
        if i in hidden_cols:
            string_file += " H "
        else:
            string_file += " r |"
    string_file += "}\n\\hline\n"
    for i in range(ncolumn):
        string_file += column_names[i] + " & "
    string_file = string_file[:-2] + " \\\\  "
    previous_data = ""
    for i in range(len(arr)):
        if previous_data != arr[i][0]:
            string_file = string_file[:-2] + " \\hline \n"
            previous_data = arr[i][0]
        elif i > 0 and arr[i - 1][1] != arr[i][1]:
            string_file = string_file[:-2] + " \\hline \n"
        # elif i % 3 == 0:
        #     string_file = string_file[:-2] + " \\hdashline \n"
        for j in range(ncolumn):
            string_file += element_to_str2(arr[i][j], arr_best[i][j]) + " & "
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


def get_best_id_n_sim_idx(a):
    best_idx = np.argmax([get_mean_and_std(b)[0] for b in a])
    n = len(a)
    ans = []
    for i in range(n):
        if i != best_idx:
            if ks_2samp(a[i], a[best_idx])[1] > 0.05:
                ans.append(i)
            # if ttest_ind(a[i], a[best_idx])[1] > 0.05:
            #     ans.append(i)
    return best_idx, ans


def add_column(table, best_id, ids):
    for j in range(len(table)):
        if j == best_id:
            table[j].append("best")
        elif j in ids:
            table[j].append("sim")
        else:
            table[j].append("non")
    return


def compare_two_pop(a, b):
    if statistics.mean(a) < statistics.mean(b):
        ans = "{ \\color{green} "
    else:
        ans = "{ \\color{black}"
    ans += "{:.2f}".format(ks_2samp(a, b)[1]) + "}"
    return ans


def compare_two_metrics(a1, b1, a2, b2):
    return compare_two_pop(a1, b1) + " " + compare_two_pop(a2, b2)


def get_idec_performance(data):
    nmi = []
    acc = []
    for seed in range(5):
        with open("../ae_experiments/model/idec_" + data + "-sdae/0.001_" + str(seed) + ".json", "r") as stat_json:
            stat = json.load(stat_json)
            s = stat[-1]
            nmi.append(s["nmi"])
            acc.append(s["acc"])
    return nmi, acc


def write_stat(filename, caption, data, arr_n, models, ref="tb-TBD", hidden_cols=[]):
    exported_table = []
    total_color_table = []
    # idec_performance = {
    #     "MNIST": [0.8661, 0.8799], "Fashion": [0.5945, 0.5163], "Reuters": [0.5310, 0.7124]
    # }
    idec_performance = {
        "MNIST": [0.8667, 0.8827], "Fashion": [0.5945, 0.5163], "Reuters": [0.5310, 0.7124]
    }
    sdae_performance = {
        "MNIST": [0.7653, 0.8270], "Fashion": [0.5842, 0.5170], "Reuters": [0.5484, 0.7371]
    }
    for i in range(len(data)):
        fig, ax = plt.subplots()
        # fig.set_size_inches(5, 7)
        xa, nmia, nmia_err, acca, acca_err = [], [], [], [], []
        xb, nmib, nmib_err, accb, accb_err = [], [], [], [], []
        xd, nmid, nmid_err, accd, accd_err = [], [], [], [], []
        idec_nmi, idec_acc = get_idec_performance(data[i])
        for n in arr_n:
            s_dict = []
            for fname, name in models.items():
                row = [data[i], n, name]
                full_name = "../test_set/Triplet-" + data[i] + "/" + str(n) + "/" + fname + ".tsv"
                if not os.path.isfile(full_name):
                    continue
                s = np.genfromtxt(full_name, delimiter="\t")
                s_dict.append(s[:, :2])
                print(full_name)
                print(s)
                ##################
                for r in range(s.shape[1]):
                    row.append(get_mean_and_std(s[:, r]))

                row.insert(5, compare_two_metrics(idec_nmi, s[:, 0], idec_acc, s[:, 1]))
                u, v = get_mean_and_std(s[:, 0])
                if "A" in fname:
                    xa.append(n)
                    nmia.append(u)
                    nmia_err.append(v)
                elif "B" in fname:
                    xb.append(n)
                    nmib.append(u)
                    nmib_err.append(v)
                else:
                    xd.append(n)
                    nmid.append(u)
                    nmid_err.append(v)
                u, v = get_mean_and_std(s[:, 1])
                if "A" in fname:
                    acca.append(u)
                    acca_err.append(v)
                elif "B" in fname:
                    accb.append(u)
                    accb_err.append(v)
                else:
                    accd.append(u)
                    accd_err.append(v)
                exported_table.append(row)
            color_table_data = []
            # print(s_dict)
            nrow = len(s_dict)
            s_dict = np.stack(s_dict)
            # print(s_dict)
            for j in range(nrow):
                row = ["", "", ""]
                color_table_data.append(row)
            nmi_arr = s_dict[:, :, 0]
            best_id, ids = get_best_id_n_sim_idx(nmi_arr)
            add_column(color_table_data, best_id, ids)

            acc_arr = s_dict[:, :, 1]
            best_id, ids = get_best_id_n_sim_idx(acc_arr)
            add_column(color_table_data, best_id, ids)
            for j in range(len(s_dict)):
                color_table_data[j].extend(["", "", "", "", ""])
            total_color_table.extend(color_table_data)
        plt.axhline(y=idec_performance[data[i]][0], c='black', linewidth=0.5, linestyle='solid', label="nmi-IDEC")
        plt.axhline(y=idec_performance[data[i]][1], c='black', linewidth=0.5, linestyle='dotted', label="acc-IDEC")
        # plt.axhline(y=sdae_performance[data[i]][0], linewidth=0.5, linestyle='solid', label="nmi-idec")
        # plt.axhline(y=sdae_performance[data[i]][1], linewidth=0.5, linestyle='dotted', label="acc-idec")

        ax.errorbar(xa, nmia, yerr=nmia_err, fmt='--v', label="nmi-EDEC-A", markersize=1, capsize=3)
        ax.errorbar(xa, acca, yerr=acca_err, fmt='--v', label="acc-EDEC-A", markersize=1, capsize=3)
        ax.errorbar(xb, nmib, yerr=nmib_err, fmt='-^', label="nmi-EDEC-B", markersize=1, capsize=3)
        ax.errorbar(xb, accb, yerr=accb_err, fmt='-^', label="acc-EDEC-B", markersize=1, capsize=3)
        ax.errorbar(xd, nmid, yerr=nmid_err, fmt=':o', label="nmi-DCC", markersize=1, capsize=3)
        ax.errorbar(xd, accd, yerr=accd_err, fmt=':o', label="acc-DCC", markersize=1, capsize=3)
        ax.set_xlabel('number of constraints')
        ax.set_ylabel('clustering performance')
        # plt.legend(bbox_to_anchor=(0.25, -0.09), loc='upper left')
        plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
        plt.tight_layout()
        plt.savefig("./triplet/triplet-" + data[i] + ".png", bbox_inches='tight')

    print(exported_table)
    with open(filename + ".tex", "w") as file:
        file.write(arr_to_table_latex(exported_table, total_color_table,
                                      ["Data", "N", "Models", "NMI", "ACC", "vsIDEC", "Avg-WMC", "\#Unsat", "\%Change",
                                       "Time (s)"], hidden_cols,
                                      caption=caption,
                                      ref=ref))


if __name__ == '__main__':
    if not os.path.exists("./triplet/"):
        os.mkdir("./triplet")
    data = ["MNIST", "Fashion", "Reuters"]
    n = [10, 100, 500, 1000]
    models = {"DCC200-pt": "DCC"}
    for fomula in ["A200", "B200"]:
        # for batch in ["64", "128", "256"]:
        for batch in ["85"]:
            # for lambda_c in ["0.01", "0.001"]:
            for lambda_c in ["0.01"]:
                # for t in ["drct", "1sdd", "nsdd"]:
                for t in ["sdd", "batch-sdd"]:
                    models[fomula + "-" + batch + "-" + t + "-" + lambda_c] = "EDEC-" + fomula[:1]
    write_stat("./triplet/triplet", "Comparison on triplet constraints with DCC and EDEC", data, n,
               models, ref="triplet", hidden_cols=[6, 8])
