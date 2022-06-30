import sys

sys.path.append("..")
from sklearn.metrics.cluster import normalized_mutual_info_score

from lib.utils import acc

from experiments.utils_setup import *
from lib.idec import IDEC
from sklearn.cluster import KMeans

if __name__ == "__main__":
    parser = data_parser("Kmeans setup")
    parser.add_argument('--pretrain', type=str, default="../model/sdae_[data]_weights.pt", metavar='N',
                        help='file path for pre-trained weights')
    args = parser.parse_args()
    if "[data]" in args.pretrain:
        args.pretrain = args.pretrain.replace("[data]", args.data)
    # Load data
    k, X, y, test_X, test_y = load_data(args.data)
    y = y.data.cpu().numpy()
    input_dim = len(X[0])
    idec = IDEC(input_dim=input_dim, z_dim=10, n_clusters=k,
                encodeLayer=[500, 500, 2000], decodeLayer=[2000, 500, 500], activation="relu", dropout=0)
    idec.load_model(args.pretrain)
    Z = idec.encodeBatch(X)
    kmeans = KMeans(k, n_init=20, random_state=0)
    idec.predict(X, y)
    kmeans.fit_predict(Z.data.cpu().numpy())
    y_pred = kmeans.labels_
    final_acc = acc(y, y_pred)
    final_nmi = normalized_mutual_info_score(y, y_pred)
    print(final_acc, final_nmi)
