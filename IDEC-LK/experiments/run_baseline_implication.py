import sys

sys.path.append("..")
from lib.complex_cs_learner import ComplexCS
from utils_setup import *
import os
from sdd_clustering.utils import *
import warnings

warnings.filterwarnings("ignore")


def reformat(pair_arr):
    if pair_arr.shape == (2,):
        return [pair_arr]
    return pair_arr


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Pairwise Constraints Program')
    # Always set to a default value = 256
    # parser.add_argument('--batch-size', type=int, default=256, metavar='N',
    #                     help='input batch size for training (default: 256)')
    parser.add_argument('--pretrain', type=str, default="./model/idec_mnist.pt", metavar='N',
                        help='directory for pre-trained weights')
    parser.add_argument('--data', type=str, default="MNIST", metavar='N', help='dataset(MNIST, Fashion, Reuters)')
    args = parser.parse_args()

    # Load data
    mnist_train = MNIST('./dataset/mnist', train=True, download=True)
    mnist_test = MNIST('./dataset/mnist', train=False)
    X = mnist_train.train_data
    y = mnist_train.train_labels
    test_X = mnist_test.test_data
    test_y = mnist_test.test_labels

    # Set parameters
    if args.data == "Fashion":
        fashionmnist_train = FashionMNIST('./dataset/fashion_mnist', train=True, download=True)
        fashionmnist_test = FashionMNIST('./dataset/fashion_mnist', train=False)
        X = fashionmnist_train.train_data
        y = fashionmnist_train.train_labels
        test_X = fashionmnist_test.test_data
        test_y = fashionmnist_test.test_labels
        args.pretrain = "./model/idec_fashion.pt"
    elif args.data == "Reuters":
        reuters_train = Reuters('./dataset/reuters', train=True, download=False)
        reuters_test = Reuters('./dataset/reuters', train=False)
        X = reuters_train.train_data
        y = reuters_train.train_labels
        test_X = reuters_test.test_data
        test_y = reuters_test.test_labels
        args.pretrain = "./model/idec_reuters.pt"
    # Print Network Structure
    # print(idec)
    y = y.data.cpu().numpy()
    # Construct constraints
    import random

    np.random.seed(1)
    random.seed(1)
    v = 4
    #v = 10
    m = 100
    folder_name = "../../SDD-for-clustering/generate_constraints/" + args.data + "-complexR2-" + str(
        v) + "-" + str(m)
    import timeit

    N = len(y)
    formu = "B"
    lambda_c = 1
    args.lr = 1
    lr2 = 1
    setup_str = "IDEC2-" + formu
    print("Training with:", setup_str)
    total_stat = []
    for test in range(5):
        # Deep logic constrained clustering
        if args.data == "Reuters":
            dlcc = ComplexCS(input_dim=2000, z_dim=10, n_clusters=4,
                             encodeLayer=[500, 500, 2000], decodeLayer=[2000, 500, 500], activation="relu", dropout=0,
                             lambda_c=lambda_c, formu=formu)
        else:
            dlcc = ComplexCS(input_dim=784, z_dim=10, n_clusters=10,
                             encodeLayer=[500, 500, 2000], decodeLayer=[2000, 500, 500], activation="relu", dropout=0,
                             lambda_c=lambda_c, formu=formu)
        dlcc.load_model(args.pretrain)
        folder_path = folder_name + "/test" + str(test).zfill(2)

        case_folder = folder_path + "/" + setup_str
        if os.path.exists(case_folder):
            # print("Setup has been run and save at:", case_folder)
            # continue
            pass
        else:
            os.makedirs(case_folder)
        arr_ids = []
        roots = []
        complex_group = []

        for group in range(m):
            ids = np.loadtxt(os.path.join(folder_path, str(group).zfill(2)) + "-indexes", dtype=int)
            mlp = np.loadtxt(os.path.join(folder_path, str(group).zfill(2)) + "-mlp", dtype=int)
            clp = np.loadtxt(os.path.join(folder_path, str(group).zfill(2)) + "-clp", dtype=int)
            mlq = np.loadtxt(os.path.join(folder_path, str(group).zfill(2)) + "-mlq", dtype=int)
            clq = np.loadtxt(os.path.join(folder_path, str(group).zfill(2)) + "-clq", dtype=int)
            mlp, clp = reformat(mlp), reformat(clp)
            mlq, clq = reformat(mlq), reformat(clq)

            arr_ids.append(ids)
            complex_group.append([mlp, clp, mlq, clq])
            if formu == "B":
                mgr, root = load_sdd_complex(folder_path + "/sdd-b/" + str(group).zfill(2))
            else:
                mgr, root = load_sdd_complex(folder_path + "/sdd-a/" + str(group).zfill(2))
            roots.append(root)
        # Train Neural Network
        stat = []
        file_prob_name = case_folder + "/prob.tsv"
        import timeit

        time_training = timeit.default_timer()
        dlcc.fit(stat, file_prob_name, arr_ids, complex_group, roots, v, X, y, case_folder + "/model.pt", lr=args.lr,
                 num_epochs=0, tol=1e-4, lr2=lr2)
        time_training = (float)(timeit.default_timer() - time_training)
        final_stat = list(stat[-1])
        final_stat.append(time_training)
        final_stat.append(test)
        total_stat.append(final_stat)
        save_tsv(case_folder + "/stat.tsv", stat)
        dlcc.save_model(case_folder + "/model.pt")
    import datetime

    dt = datetime.datetime.now()
    seq = str(int(dt.strftime("%d%H%M%S")))
    save_tsv(folder_name + "/" + setup_str + "-" + seq + ".tsv", total_stat)
