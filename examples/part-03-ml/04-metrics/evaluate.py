from sklearn.metrics import precision_recall_fscore_support, confusion_matrix
y_true = [0, 1, 1, 0, 1, 1, 0, 0]
y_pred = [0, 1, 0, 0, 1, 1, 0, 1]
p, r, f1, _ = precision_recall_fscore_support(y_true, y_pred, average="binary")
print("P/R/F1:", round(p,2), round(r,2), round(f1,2))
print("混淆矩阵:\n", confusion_matrix(y_true, y_pred))
