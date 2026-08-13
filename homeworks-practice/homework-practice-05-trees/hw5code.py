import numpy as np
from collections import Counter


def find_best_split(feature_vector, target_vector, min_samples_leaf=None):
    """
    Под критерием Джини здесь подразумевается следующая функция:
    $$Q(R) = -\frac {|R_l|}{|R|}H(R_l) -\frac {|R_r|}{|R|}H(R_r)$$,
    $R$ — множество объектов, $R_l$ и $R_r$ — объекты, попавшие в левое и правое поддерево,
     $H(R) = 1-p_1^2-p_0^2$, $p_1$, $p_0$ — доля объектов класса 1 и 0 соответственно.

    Указания:
    * Пороги, приводящие к попаданию в одно из поддеревьев пустого множества объектов, не рассматриваются.
    * В качестве порогов, нужно брать среднее двух сосдених (при сортировке) значений признака
    * Поведение функции в случае константного признака может быть любым.
    * При одинаковых приростах Джини нужно выбирать минимальный сплит.
    * За наличие в функции циклов балл будет снижен. Векторизуйте! :)

    :param feature_vector: вещественнозначный вектор значений признака
    :param target_vector: вектор классов объектов,  len(feature_vector) == len(target_vector)

    :return thresholds: отсортированный по возрастанию вектор со всеми возможными порогами, по которым объекты можно
     разделить на две различные подвыборки, или поддерева
    :return ginis: вектор со значениями критерия Джини для каждого из порогов в thresholds len(ginis) == len(thresholds)
    :return threshold_best: оптимальный порог (число)
    :return gini_best: оптимальное значение критерия Джини (число)
    """
    if len(np.unique(feature_vector)) == 1:
        return np.array([]), np.array([]), None, None

    sorted_features = np.unique(sorted(feature_vector))
    thresholds = (sorted_features[1:] + sorted_features[:-1]) / 2

    def Q(x):
        p1l = np.sum((feature_vector < x) & (target_vector == 1)) / np.sum(feature_vector < x)
        p0l = np.sum((feature_vector < x) & (target_vector == 0)) / np.sum(feature_vector < x)
        p1r = np.sum((feature_vector > x) & (target_vector == 1)) / np.sum(feature_vector > x)
        p0r = np.sum((feature_vector > x) & (target_vector == 0)) / np.sum(feature_vector > x)
        if p1l > 1:
            raise ValueError(f"p1l is {p1l}")
        if p1r > 1:
            raise ValueError(f"p1r is {p1r}")
        if p0l > 1:
            raise ValueError(f"p0l is {p0l}")
        if p0r > 1:
            raise ValueError(f"p0r is {p0r}")
        hl = 1 - p0l**2 - p1l**2
        hr = 1 - p0r**2 - p1r**2
        rl = np.sum(feature_vector < x)
        rr = np.sum(feature_vector > x)
        return - (rl * hl) / (rl + rr) - (rr * hr) / (rl + rr)

    ginis = []
    for threshold in thresholds:
        cnt = np.sum(feature_vector < threshold)
        if min_samples_leaf is not None and min(cnt, len(feature_vector) - cnt) < min_samples_leaf:
            ginis.append(None)
            continue
        ginis.append(Q(threshold))

    best_threshold = None
    best_gini = None
    for i in range(len(ginis)):
        if best_gini is None or (ginis[i] is not None and ginis[i] > best_gini):
            best_gini = ginis[i]
            best_threshold = thresholds[i]
    return thresholds, ginis, best_threshold, best_gini





class DecisionTree:
    def __init__(self, feature_types, max_depth=None, min_samples_split=None, min_samples_leaf=None):
        if np.any(list(map(lambda x: x != "real" and x != "categorical", feature_types))):
            raise ValueError("There is unknown feature type")

        self._tree = {}
        self._feature_types = feature_types
        self._max_depth = max_depth
        self._min_samples_split = min_samples_split
        self._min_samples_leaf = min_samples_leaf

    def _fit_node(self, sub_X, sub_y, node, depth):
        if np.all(sub_y == sub_y[0]):
            node["type"] = "terminal"
            node["class"] = sub_y[0]
            return

        if self._max_depth is not None and depth >= self._max_depth:
            node["type"] = "terminal"
            node["class"] = Counter(sub_y).most_common(1)[0][0]
            return

        if self._min_samples_split is not None and len(sub_y) < self._min_samples_split:
            node["type"] = "terminal"
            node["class"] = Counter(sub_y).most_common(1)[0][0]
            return

        feature_best, threshold_best, gini_best, split = None, None, None, None
        for feature in range(sub_X.shape[1]):
            feature_type = self._feature_types[feature]
            categories_map = {}

            if feature_type == "real":
                feature_vector = sub_X[:, feature]
            elif feature_type == "categorical":
                counts = Counter(sub_X[:, feature])
                clicks = Counter(sub_X[sub_y == 1, feature])
                ratio = {}
                for key, current_count in counts.items():
                    if key in clicks:
                        current_click = clicks[key]
                    else:
                        current_click = 0
                    ratio[key] = current_click / current_count
                sorted_categories = list(map(lambda x: x[0], sorted(ratio.items(), key=lambda x: x[1])))
                categories_map = dict(zip(sorted_categories, list(range(len(sorted_categories)))))

                feature_vector = np.array(list(map(lambda x: categories_map[x], sub_X[:, feature])))
            else:
                raise ValueError

            if len(feature_vector) <= 3:
                continue

            _, _, threshold, gini = find_best_split(feature_vector, sub_y, self._min_samples_leaf)
            if (gini is not None) and (gini_best is None or gini > gini_best):
                feature_best = feature
                gini_best = gini
                split = feature_vector < threshold

                if feature_type == "real":
                    threshold_best = threshold
                elif feature_type == "categorical":
                    threshold_best = list(map(lambda x: x[0],
                                              filter(lambda x: x[1] < threshold, categories_map.items())))
                else:
                    raise ValueError

        if feature_best is None:
            node["type"] = "terminal"
            node["class"] = Counter(sub_y).most_common(1)[0][0]
            return

        node["type"] = "nonterminal"

        node["feature_split"] = feature_best
        if self._feature_types[feature_best] == "real":
            node["threshold"] = threshold_best
        elif self._feature_types[feature_best] == "categorical":
            node["categories_split"] = threshold_best
        else:
            raise ValueError
        node["left_child"], node["right_child"] = {}, {}
        self._fit_node(sub_X[split], sub_y[split], node["left_child"], depth + 1)
        self._fit_node(sub_X[np.logical_not(split)], sub_y[np.logical_not(split)], node["right_child"], depth + 1)

    def _predict_node(self, x, node):
        if node["type"] == "terminal":
            return node["class"]
        feature_split = node["feature_split"]
        split_feature_value = x[feature_split]
        if self._feature_types[feature_split] == "real":
            if split_feature_value < node["threshold"]:
                return self._predict_node(x, node["left_child"])
            else:
                return self._predict_node(x, node["right_child"])
        elif self._feature_types[feature_split] == "categorical":
            if split_feature_value in node["categories_split"]:
                return self._predict_node(x, node["left_child"])
            else:
                return self._predict_node(x, node["right_child"])
        else:
            raise ValueError



    def fit(self, X, y):
        self._fit_node(X, y, self._tree, 0)

    def predict(self, X):
        predicted = []
        for x in X:
            predicted.append(self._predict_node(x, self._tree))
        return np.array(predicted)
