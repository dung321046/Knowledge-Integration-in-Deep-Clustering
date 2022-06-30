import torch
class ProbCalculator:

    def __init__(self, probs):
        self.probs = probs
        self.cal_nodes = dict()

    def calculate_re(self, node):
        if node.id in self.cal_nodes:
            return self.cal_nodes[node.id]
        if node.node_size() == 0:
            if node.is_true():
                return 1.0
            if node.literal > 0:
                return self.probs[node.literal - 1]
            return 1.0 - self.probs[- node.literal - 1]
        t = node.elements()
        ans = 0.0
        for i in range(len(t)):
            if t[i][1].is_false():
                continue
            u = self.calculate_re(t[i][0])
            v = self.calculate_re(t[i][1])
            ans += u * v
        self.cal_nodes[node.id] = ans
        return ans

    def calculate(self, node):
        if node.is_false():
            return 0.0
        if node.is_true():
            return 1.0
        return self.calculate_re(node)


def weight_convert_b(probs, n, k):
    p = torch.ones(n, k)
    for i in range(n):
        for j in range(k - 1):
            denominator = 1.0 - sum(probs[i][:j])
            p[i][j] = probs[i][j] / denominator
    p = p.flatten()
    return p


def weight_convert_b_batch(probs, batch_size, n, k):
    p_matrix = []
    probs = torch.stack(probs)
    for b in range(batch_size):
        p = torch.ones(n, k)
        for i in range(n):
            for j in range(k - 1):
                denominator = 1.0 - sum(probs[i * k:i * k + j, b])
                p[i][j] = probs[i * k + j][b] / denominator
        p_matrix.append(p.flatten())
    return torch.transpose(torch.stack(p_matrix), 0, 1)

