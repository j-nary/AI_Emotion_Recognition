import cv2
import torch
from PIL import Image
from torchvision import transforms
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import confusion_matrix, accuracy_score, precision_score, recall_score, f1_score

model = torch.hub.load('ultralytics/yolov5', 'custom', path='./yolov5-master/runs/train/feeling/weights/best.pt', force_reload=True)

image_paths = ['./test_image/test1.jpeg', './test_image/test2.jpeg', './test_image/test3.jpeg', './test_image/test4.jpeg', './test_image/test5.jpeg']

y = []
for image in image_paths:
    results = model(image)
    results = results.xyxy[0]
    y.append(results[:, -1])

y_pred = []
for pred in y:
    if len(pred) == 0:
        y_pred.append(2)
    else:
        y_pred.append(int(pred[-1]))
y_true = [0, 2, 1, 2, 1]

# 성능 지표 계산
accuracy = accuracy_score(y_true, y_pred)
precision = precision_score(y_true, y_pred, average='macro')
recall = recall_score(y_true, y_pred, average='macro')
f1 = f1_score(y_true, y_pred, average='macro')

# 성능 지표 라벨과 값
metric_labels = ['accuracy', 'precision', 'recall', 'F1 score']
metrics = np.array([accuracy, precision, recall, f1])

conf_mat = confusion_matrix(y_true, y_pred)
classes = ['good', 'bad', 'nothing']

# 혼동 행렬을 이미지로 변환
plt.figure(figsize=(10, 7))
sns.heatmap(conf_mat, annot=True, fmt='d', xticklabels=classes, yticklabels=classes, cmap='Blues')
plt.xlabel('predict label')
plt.ylabel('reality label')
plt.title('Confusion Matrix for Emotion Recognition Results')
conf_mat_img = 'confusion_matrix.png'
plt.savefig(conf_mat_img)
plt.close()

# 성능 지표를 이미지로 변환
plt.figure(figsize=(8, 5))
sns.barplot(x=metric_labels, y=metrics)
plt.title('Model Performance Evaluation Metrics')
performance_metrics_img = 'performance_metrics.png'
plt.savefig(performance_metrics_img)
plt.close()

# OpenCV를 사용하여 이미지 표시
conf_mat_image = cv2.imread(conf_mat_img)
cv2.imshow('Confusion Matrix', conf_mat_image)

performance_metrics_image = cv2.imread(performance_metrics_img)
cv2.imshow('Performance Metrics', performance_metrics_image)

cv2.waitKey(0)
cv2.destroyAllWindows()
