import numpy as np
from sklearn.datasets import load_iris

class DecisionTree(object):
    """
    基于ID3生成算法
    """
    def __init__(self, alg = 'ID3'): # 构造方法,创建实例时自动调用
        self.rate, self.root = None, None
        self.alg = alg
      
    def fit(self, X: np.ndarray, y: np.ndarray, rate: float = 0.95):
        """
        - rate: 叶节点的最小纯度阈值。
        """
        self.rate = rate
        self.root = self.build_tree(X, y, np.arange(X.shape[0]), np.arange(X.shape[1]))

    def __call__(self, X: np.ndarray):
        """
        预测结果：每个输入实例的预测标签的数组。
        """
        return np.array([self.predict(self.root, x) for x in X])

    def predict(self, node, x: np.ndarray):
        """
        递归地使用决策树对给定输入实例进行预测。

        参数：
        - node: 当前正在评估的节点。
        - x: 要进行预测的输入实例。

        返回：
        - 预测结果：输入实例的预测标签。
        """
        if isinstance(node, dict):  # 如果节点是树（字典）类型
            col, trees = node["col"], node["trees"]
            # 基于值进行递归预测
            return self.predict(trees[x[col]], x)
        return node  # 如果节点是叶节点，直接返回其值

    def build_tree(self, X: np.ndarray, y: np.ndarray, rows: np.ndarray, cols: np.ndarray):
        """
        递归构建决策树。
        参数：
        - rows: 与当前节点相关的数据点的索引。
        - cols: 可用于分裂的特征。
        返回：
        - node: 构建的决策树节点（可以是字典或叶节点值）。
        """
        cats = np.bincount(y[rows]) # 统计每个类别的数量

        # 如果没有剩余特征或节点满足纯度条件
        if len(cols) == 0 or np.max(cats) / len(rows) > self.rate:
            return np.argmax(cats)  # 返回出现频率最高的类别

        # ID3 信息增益
        if self.alg == 'ID3':
            # 选择最佳特征
            k = np.argmax([self.calc_info_gain(X, y, rows, f) for f in cols])
        # C4.5 信息增益率
        elif self.alg == 'C4.5':
            # 选择最佳特征
            k = np.argmax([self.calc_info_gain(X, y, rows, f) / self.calc_exp_ent(y, rows) for f in cols])
        # CART 基尼指数
        elif self.alg == 'CART':
            # 选择最佳特征
            k = np.argmax([self.calc_gini_index(X, y, rows, f) for f in cols])

        col = cols[k]
        cols = np.delete(cols, k)  # 移除选择的特征

        # 为选择的特征创建子树
        trees = {
            value: self.build_tree(X, y, rows[X[rows, col] == value], cols)
            for value in np.unique(X[rows, col]).tolist()  # 为特征的每个唯一值创建一个子树
        }
        return {"col": col, "trees": trees}
        # TODO(tao): 后剪枝6


    @staticmethod
    def calc_exp_ent(y: np.ndarray, rows: np.ndarray):  # 计算经验熵
        prob = np.bincount(y[rows]) / len(rows)
        prob = prob[prob.nonzero()]  # 除去0概率
        return np.sum(-prob * np.log(prob))  # 经验熵

    @classmethod
    def calc_cnd_ent(cls, X: np.ndarray, y: np.ndarray, rows: np.ndarray, col: int):  # 计算条件熵
        ent = 0  # 经验条件熵
        for value in np.unique(X[rows, col]):
            indices_ = rows[X[rows, col] == value]
            ent += len(indices_) / len(rows) * cls.calc_exp_ent(y, indices_)
        return ent  # 条件熵

    @classmethod
    def calc_info_gain(cls, X: np.ndarray, y: np.ndarray, rows: np.ndarray, col: int):  # 计算信息增益
        exp_ent = cls.calc_exp_ent(y, rows)  # 经验熵
        cnd_ent = cls.calc_cnd_ent(X, y, rows, col)  # 经验条件熵
        return exp_ent - cnd_ent  # 信息增益

    @classmethod
    def calc_gini_index(cls, X: np.ndarray, y: np.ndarray, rows: np.ndarray, col: int): 
        # 计算基尼指数
        gini_index = 0
        for value in np.unique(X[rows, col]):
            indices_ = rows[X[rows, col] == value]
            gini_index += len(indices_) / len(rows) * cls.calc_gini(y, indices_)
        return gini_index

    @staticmethod
    def calc_gini(y: np.ndarray, rows: np.ndarray):
        prob = np.bincount(y[rows]) / len(rows)
        return 1 - np.sum(prob**2)


if __name__ == "__main__":

    data = load_iris() 
    X = data['data']
    y = data['target']
    decision_tree = DecisionTree(alg='CART')
    decision_tree.fit(X, y, rate=0.70)
    y_pred = decision_tree(X)

    print(decision_tree.root)
    print(y)
    print(y_pred)

    acc = np.sum(y_pred == y) / len(y_pred)
    print(f"Accuracy = {100 * acc:.2f}%")
