import numpy as np
import tensorflow as tf
from PIL import Image
import os

model = tf.keras.models.load_model(r"E:\olive_output3\olive_model_final.keras")
class_names = ['Healthy', 'aculus_olearius', 'olive_peacock_spot']

def eval_class(folder, true_label):
    correct = 0
    total = 0
    for fname in os.listdir(folder):
        path = os.path.join(folder, fname)
        try:
            img = Image.open(path).convert("RGB").resize((224, 224))
            arr = np.expand_dims(np.array(img, dtype=np.float32), 0)
            out = model.predict(arr, verbose=0)[0]
            if class_names[out.argmax()] == true_label:
                correct += 1
            total += 1
        except:
            pass
    print(f"{true_label}: {correct}/{total} = {correct/total*100:.1f}%")

eval_class(r"E:\dataset2\test\Healthy", "Healthy")
eval_class(r"E:\dataset2\test\aculus_olearius", "aculus_olearius")
eval_class(r"E:\dataset2\test\olive_peacock_spot", "olive_peacock_spot")