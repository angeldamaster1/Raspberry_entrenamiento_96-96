import os
import cv2
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import tensorflow as tf
import time
from sklearn.metrics import confusion_matrix, classification_report, ConfusionMatrixDisplay

# === CONFIGURACIÓN ===
TFLITE_MODEL_PATH = 'modelo_cuantizado_INT8_3.tflite'
CARPETAS = {
    'good_quality': 1,   # Clase 1 → Sano
    'bad_quality': 0     # Clase 0 → Desecho
}
IMG_SIZE = (96, 96)
CSV_RESULTADOS = 'resultados_predicciones.csv'
IMG_MATRIZ = 'matriz_confusion.jpg'
TXT_METRICAS = 'metricas_modelo.txt'
CLASES = {0: 'Desecho', 1: 'Sano'}

# === CARGAR MODELO TFLITE ===
interpreter = tf.lite.Interpreter(model_path=TFLITE_MODEL_PATH)
interpreter.allocate_tensors()
input_details = interpreter.get_input_details()
output_details = interpreter.get_output_details()
input_dtype = input_details[0]['dtype']

# === RESULTADOS ===
resultados = []
y_true = []
y_pred = []

tiempo_total_inicio = time.perf_counter()

# === PROCESAR TODAS LAS CARPETAS ===
for carpeta, etiqueta_real in CARPETAS.items():
    for nombre_img in os.listdir(carpeta):
        if not nombre_img.lower().endswith(('.jpg', '.jpeg', '.png')):
            continue

        ruta = os.path.join(carpeta, nombre_img)
        img = cv2.imread(ruta)
        if img is None:
            print(f"❌ No se pudo leer: {nombre_img}")
            continue

        # Preprocesamiento
        img_gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        img_resized = cv2.resize(img_gray, IMG_SIZE)
        img_input = np.expand_dims(img_resized, axis=-1)
        img_input = np.expand_dims(img_input, axis=0)

        if input_dtype == np.uint8:
            img_input = img_input.astype(np.uint8)
        else:
            img_input = (img_input / 255.0).astype(np.float32)

        # Inferencia
        interpreter.set_tensor(input_details[0]['index'], img_input)
        interpreter.invoke()
        output = interpreter.get_tensor(output_details[0]['index'])[0][0]
        pred = 1 if output >= 0.5 else 0

        resultados.append({
            'imagen': nombre_img,
            'carpeta': carpeta,
            'probabilidad_sano': float(output),
            'prediccion': pred,
            'clase_predicha': CLASES[pred],
            'etiqueta_real': etiqueta_real,
            'clase_real': CLASES[etiqueta_real]
        })

        y_true.append(etiqueta_real)
        y_pred.append(pred)

tiempo_total_fin = time.perf_counter()
print(f"🕒 Tiempo total: {(tiempo_total_fin - tiempo_total_inicio):.2f} s")

# === GUARDAR CSV ===
df_resultados = pd.DataFrame(resultados)
df_resultados.to_csv(CSV_RESULTADOS, index=False)
print(f"✅ CSV guardado en: {CSV_RESULTADOS}")

# === MATRIZ DE CONFUSIÓN ===
cm = confusion_matrix(y_true, y_pred)
disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=['Desecho', 'Sano'])

plt.figure(figsize=(5, 5))
disp.plot(cmap='Blues', values_format='d')
plt.title("Matriz de Confusión")
plt.savefig(IMG_MATRIZ)
plt.close()
print(f"🖼️ Matriz de confusión guardada en: {IMG_MATRIZ}")

# === MÉTRICAS ===
reporte = classification_report(y_true, y_pred, target_names=['Desecho', 'Sano'], digits=4)
print("\n📊 Reporte de métricas:\n", reporte)

# Guardar métricas en archivo .txt
with open(TXT_METRICAS, 'w') as f:
    f.write("== MÉTRICAS DEL MODELO ==\n\n")
    f.write(reporte)
print(f"📝 Métricas guardadas en: {TXT_METRICAS}")
