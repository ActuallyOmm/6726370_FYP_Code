# @title Metrics
from sklearn import metrics
from sklearn.metrics import confusion_matrix, classification_report
import seaborn as sns
import matplotlib.pyplot as plt
import torch

def confusionMatrixPlot(y_true, y_pred, name):
    # Resource used to create confusion matrix:
    # https://seaborn.pydata.org/generated/seaborn.heatmap.html
    class_names = ['Anger','Disgust','Fear','Happiness','Sadness','Surprise','Neurtal']
    conf = confusion_matrix(y_true, y_pred)
    cm = sns.heatmap(conf, annot=True, fmt=".0f")
    cm.set_xticklabels(class_names, fontsize=12, rotation=45)
    cm.set_yticklabels(class_names, fontsize=12, rotation=45)
    plt.xlabel("Predicted Labels")
    plt.ylabel("True Labels")
    plt.title("Confusion Matrix" + str(name))
    plt.show()

    # Resource to generate classification report
    # https://scikit-learn.org/1.5/modules/generated/sklearn.metrics.classification_report.html
    class_report = classification_report(y_true, y_pred, target_names=class_names, digits=4)
    print(class_report)

def lossPlot(train_losses, val_losses, name):
    plt.plot(train_losses, label='Training Loss')
    plt.plot(val_losses, label='Validation Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.title('Training and Validation Loss Curves ' + str(name))
    plt.legend()
    plt.show()

def accuracyPlot(train_accuracy, val_accuracy, name):
    plt.plot(train_accuracy, label='Training Acc')
    plt.plot(val_accuracy, label='Validation Acc')
    plt.xlabel('Epoch')
    plt.ylabel('Accuracy')
    plt.title('Training and Validation Acc Curves'+ str(name))
    plt.legend()
    plt.show()
