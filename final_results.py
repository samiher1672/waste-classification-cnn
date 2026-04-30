import tensorflow as tf
import numpy as np
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from sklearn.metrics import classification_report, confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns

model = tf.keras.models.load_model("waste_model.h5")
class_names = ['glass', 'organic', 'paper']

test_datagen = ImageDataGenerator(rescale=1./255)
test_generator = test_datagen.flow_from_directory(
    "dataset/test",
    target_size=(150, 150),
    batch_size=32,
    class_mode='categorical',
    shuffle=False
)

predictions = model.predict(test_generator, verbose=0)
predicted_classes = np.argmax(predictions, axis=1)
true_classes = test_generator.classes

test_loss, test_accuracy = model.evaluate(test_generator, verbose=0)
cm = confusion_matrix(true_classes, predicted_classes)

fig = plt.figure(figsize=(14, 10))

fig.suptitle(f'WASTE CLASSIFICATION MODEL - FINAL RESULTS\nTest Accuracy: {test_accuracy:.2%}',
             fontsize=18, fontweight='bold', color='darkgreen')

ax1 = fig.add_subplot(2, 2, 1)
sns.heatmap(cm, annot=True, fmt='d', cmap='Greens',
            xticklabels=class_names, yticklabels=class_names, ax=ax1)
ax1.set_title('Confusion Matrix', fontsize=14)
ax1.set_xlabel('Predicted')
ax1.set_ylabel('True')

ax2 = fig.add_subplot(2, 2, 2)
class_acc = cm.diagonal() / cm.sum(axis=1) * 100
bars = ax2.bar(class_names, class_acc, color=['#2ecc71', '#27ae60', '#1abc9c'])
ax2.set_ylim(0, 100)
ax2.set_title('Accuracy Per Class', fontsize=14)
ax2.set_ylabel('Accuracy (%)')
for bar, acc in zip(bars, class_acc):
    ax2.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 1,
             f'{acc:.1f}%', ha='center', fontweight='bold', fontsize=12)

ax3 = fig.add_subplot(2, 1, 2)
report = classification_report(true_classes, predicted_classes,
                                target_names=class_names, output_dict=True)
table_data = []
for cls in class_names:
    table_data.append([cls,
                       f'{report[cls]["precision"]:.1%}',
                       f'{report[cls]["recall"]:.1%}',
                       f'{report[cls]["f1-score"]:.1%}'])
table_data.append(['AVERAGE',
                   f'{report["macro avg"]["precision"]:.1%}',
                   f'{report["macro avg"]["recall"]:.1%}',
                   f'{report["macro avg"]["f1-score"]:.1%}'])

table = ax3.table(cellText=table_data,
                  colLabels=['Class', 'Precision', 'Recall', 'F1-Score'],
                  cellLoc='center', loc='center')
table.auto_set_font_size(False)
table.set_fontsize(12)
table.scale(1, 2)
ax3.axis('off')
ax3.set_title('Classification Report', fontsize=14, pad=20)

plt.tight_layout()
plt.savefig('final_results.png', dpi=150, bbox_inches='tight')
plt.show()

print(f"\n{'='*40}")
print(f"FINAL TEST ACCURACY: {test_accuracy:.2%}")
print(f"{'='*40}")
print("\nPer-Class Accuracy:")
for name, acc in zip(class_names, class_acc):
    print(f"  {name}: {acc:.1f}%")