import os


class Node:
    def __init__(self, id, literal, size):
        self.id = id
        self.literal = literal
        self.children = []
        self.children_id = []
        self.size = size

    def node_size(self):
        return self.size

    def elements(self):
        ans = []
        for i in range(int(len(self.children) / 2)):
            ans.append([self.children[i * 2], self.children[i * 2 + 1]])
        return ans

    def is_false(self):
        return False

    def is_true(self):
        return self.literal == 0


class ArithmeticLoader:

    def __init__(self):
        self.node_info = dict()

    def load_node(self, node):
        if node.id in self.node_info:
            return
        if node.node_size() == 0:
            self.node_info[node.id] = {"literal": node.literal, "children": [], "types": []}
            return
        t = node.elements()
        self.node_info[node.id] = {"literal": -1, "children": [], "types": []}
        for i in range(len(t)):
            # For sdd, second term is never happen. But we did for general trees
            if t[i][1].is_false() or t[i][0].is_false():
                continue
            self.node_info[node.id]["children"].append(t[i][0].id)
            self.load_node(t[i][0])
            self.node_info[node.id]["children"].append(t[i][1].id)
            self.load_node(t[i][1])
        return

    def to_text(self, root):
        self.node_info = dict()
        self.load_node(root)
        out_str = str(root.id) + "\n"
        for node_id, node_data in self.node_info.items():
            out_str += str(node_id) + " " + str(node_data["literal"])
            for i in range(len(node_data["children"])):
                out_str += " " + str(node_data["children"][i])  # + " " + str(node_data["types"][i])
            out_str += "\n"
        return out_str


def save_t(f, filename):
    al = ArithmeticLoader()
    with open(filename, "w") as file:
        file.write(al.to_text(f))
    # print(al.to_text(f))


def load_t(filename):
    node_info = dict()
    with open(filename, "r") as file:
        lines = file.readlines()
        root = int(lines[0])
        for i in range(1, len(lines)):
            numbers = [int(x) for x in lines[i].split()]
            num_child = len(numbers) - 2
            node = Node(numbers[0], numbers[1], num_child)
            for j in range(num_child):
                node.children_id.append(numbers[j + 2])
            node_info[numbers[0]] = node
    for id, node in node_info.items():
        for child_id in node.children_id:
            if child_id not in node_info:
                print("Error", child_id)
            else:
                node.children.append(node_info[child_id])
    return node_info[root]


def save_model(path, name, mgr, root, saveSDD=True, saveT=True):
    if not os.path.exists(path):
        os.makedirs(path)
    if saveSDD:
        vtree = mgr.vtree()
        vtree.save(bytes(os.path.join(path, name + ".vtree"), encoding="utf8"))
        mgr.save(bytes(os.path.join(path, name + ".sdd"), encoding="utf8"), root)
    if saveT:
        save_t(root, os.path.join(path, name + ".txt"))
    return


def load_pw_sdd(model, k):
    ml_prefix = "../sdd/ml" + model + "/ml" + model + "-" + str(k)
    cl_prefix = "../sdd/cl" + model + "/cl" + model + "-" + str(k)

    # ml_vtree = Vtree.from_file(bytes(ml_prefix + ".vtree", encoding="utf8"))
    # ml_mgr = SddManager.from_vtree(ml_vtree)
    # ml_root = ml_mgr.read_sdd_file(bytes(ml_prefix + ".sdd", encoding="utf8"))
    #
    # cl_vtree = Vtree.from_file(bytes(cl_prefix + ".vtree", encoding="utf8"))
    # cl_mgr = SddManager.from_vtree(cl_vtree)
    # cl_root = ml_mgr.read_sdd_file(bytes(cl_prefix + ".sdd", encoding="utf8"))
    ml_root = load_t(ml_prefix + ".txt")
    cl_root = load_t(cl_prefix + ".txt")
    return ml_root, cl_root


def load_triplet_sdd(model, k):
    file_path = "../sdd/triplet" + model + "/triplet" + model + "-" + str(k) + ".txt"
    root = load_t(file_path)
    return root
