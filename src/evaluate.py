from sklearn.metrics import classification_report, confusion_matrix, roc_auc_score
from src.train import train_model


def evaluate_model():
    pipeline, X_test, y_test = train_model()

    y_pred = pipeline.predict(X_test)
    y_prob = pipeline.predict_proba(X_test)[:, 1]

    print("\n📊 Confusion Matrix:")
    print(confusion_matrix(y_test, y_pred))

    print("\n📄 Classification Report:")
    print(classification_report(y_test, y_pred))

    print("\n🎯 ROC-AUC Score:")
    print(roc_auc_score(y_test, y_prob))


if __name__ == "__main__":
    evaluate_model()