import sys

sys.path.append("..")

from sklearn.metrics.cluster import normalized_mutual_info_score

from lib.utils import acc

from experiments.utils_setup import *
from sklearn.cluster import KMeans
import statistics
import numpy as np

INF = 1000000
from lib.idec import IDEC


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
    parser = clustering_parser("Improved Deep Clustering")
    parser.add_argument('--lr', type=float, default=0.0001, metavar='N',
                        help='learning rate for training (default: 0.001)')
    args = parser.parse_args()
    # Load data
    k, X, y, test_X, test_y = load_data(args.data)
    y = y.data.cpu().numpy()
    input_dim = len(X[0])
    exported_table = []
    nmis = []
    accs = []
    for seed in range(5):
        idec = IDEC(input_dim=input_dim, z_dim=10, n_clusters=k,
                    encodeLayer=[500, 500, 2000], decodeLayer=[2000, 500, 500], activation="relu", dropout=0)
        idec.load_model("./model/sdae_" + args.data + "/" + str(seed) + ".pt")
        row = [args.data, seed]
        kmeans = KMeans(k, n_init=20, random_state=0)
        Z = idec.encodeBatch(X)
        kmeans.fit_predict(Z.data.cpu().numpy())
        y_pred = kmeans.labels_
        final_acc = acc(y, y_pred)
        final_nmi = normalized_mutual_info_score(y, y_pred)
        row.append(final_nmi)
        row.append(final_acc)
        nmis.append(final_nmi)
        accs.append(final_acc)
        exported_table.append(row)
    row = [args.data, "Average"]
    row.append(get_mean_and_std(nmis))
    row.append(get_mean_and_std(accs))
    exported_table.append(row)
    with open(args.data + "-sdae-raw.tex", "w") as file:
        file.write(arr_to_table_latex(exported_table, ["Data", "Run", "NMI", "ACC"], [],
                                      caption="Raw training results on " + args.data + " with SDAE + Kmeans",
                                      ref="idec-raw-" + args.data))
