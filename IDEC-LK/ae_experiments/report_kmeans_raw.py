import sys

sys.path.append("..")
from sklearn.metrics.cluster import normalized_mutual_info_score

from lib.utils import acc

from experiments.utils_setup import *
from sklearn.cluster import KMeans

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
    string_file += "\\centering\n\\caption{" + caption + "}\\label{tab:" + ref + "}" + "\n\\begin{tabular}{ |"
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


if __name__ == "__main__":
    exported_table = []
    for data in ["MNIST", "Fashion", "Reuters"]:
        k, X, y, test_X, test_y = load_data(data)
        y = y.data.cpu().numpy()
        input_dim = len(X[0])
        kmeans = KMeans(k, n_init=20, random_state=0)
        kmeans.fit_predict(X.data.cpu().numpy())
        y_pred = kmeans.labels_
        final_acc = acc(y, y_pred)
        final_nmi = normalized_mutual_info_score(y, y_pred)

        print(final_acc, final_nmi)
        exported_table.append([data, final_nmi, final_acc])
    with open("kmeans-raw.tex", "w") as file:
        file.write(arr_to_table_latex(exported_table, ["Data", "NMI", "ACC"], [],
                                      caption="Clustering performance with Kmeans", ref="kmeans-raw"))
