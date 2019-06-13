from sklearn.metrics import precision_recall_curve, auc, roc_auc_score


class Evaluator:
    def pr_auc_score(self, y_true, y_pred):
        precision, recall, _ = precision_recall_curve(y_true, y_pred)
        pr_auc = auc(recall, precision)
        return pr_auc

    def roc_auc_score(self, y_true, y_pred):
        return roc_auc_score(y_true, y_pred)
