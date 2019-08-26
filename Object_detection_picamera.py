#---------- Deteccion de objetos con PiCamera usando Tensorflow ----------
#
# Luis Arias Gomez, Edward Umana Williams, Guillermo Lopez Navarro
# Reconocimiento de Patrones - Sistemas embebidos de Alto Desempeno
# Tecnologico de Costa Rica
# Agosto 2019

# Este programa utiliza un clasificador con TensorFlow para deteccion de objetos.
# Se carga un modelo entrenado al cual se le brinda imagenes capturadas con la 
#    camara, lo cual retorna una etiqueta que posteriormente se utiliza en conjunto
#    con un marco que rodea el objeto, para denotar su identidad segun lo predijo
#    el modelo.

# Este codigo se basa parcialmente en el ejemplo de Google disponible en
# https://github.com/tensorflow/models/blob/master/research/object_detection/object_detection_tutorial.ipynb

# Asimismo, se utilizo la guia disponible en el link a continuacion, para la 
#     integracion de la camara RPI con la Jetson Nano.
# https://www.jetsonhacks.com/2019/04/02/jetson-nano-raspberry-pi-camera/

import os
import cv2
import numpy as np
import tensorflow as tf

# Utilities para etiquetado en el video a mostrar y para manipulacion de
#     las etiquetas del modelo, respectivamente
from utils import visualization_utils as vis_util
from utils import label_map_util

# Dimensiones del video a mostrar
IM_WIDTH = 1280#720 #1280 # 640
IM_HEIGHT = 720#540 #720 # 480

# Funcion que retorna handler de la camara RPI
def gstreamer_pipeline (capture_width=IM_WIDTH, capture_height=IM_HEIGHT, display_width=IM_WIDTH, display_height=IM_HEIGHT, framerate=60, flip_method=0) :   
    return ('nvarguscamerasrc ! ' 
    'video/x-raw(memory:NVMM), '
    'width=(int)%d, height=(int)%d, '
    'format=(string)NV12, framerate=(fraction)%d/1 ! '
    'nvvidconv flip-method=%d ! '
    'video/x-raw, width=(int)%d, height=(int)%d, format=(string)BGRx ! '
    'videoconvert ! '
    'video/x-raw, format=(string)BGR ! appsink'  % (capture_width,capture_height,framerate,flip_method,display_width,display_height))

# Nombre del directorio que contiene el modelo a utilizar para la prediccion
#MODEL_NAME = 'ssdlite_mobilenet_v2_coco_2018_05_09'
MODEL_NAME = 'birras'
#LABELS = 'mscoco_label_map.pbtxt'
#LABELS = 'birras_labelmap_6.pbtxt'
LABELS = 'birras_labelmap.pbtxt'

# Numero de clases que puede identificar el modelo
#NUM_CLASSES = 90
#NUM_CLASSES = 6
NUM_CLASSES = 2

# Directorio actual
CWD_PATH = os.getcwd()

# Ruta al archivo .pb (modelo a utilizar)
PATH_TO_CKPT = os.path.join(CWD_PATH, MODEL_NAME, 'frozen_inference_graph.pb')
#PATH_TO_CKPT = os.path.join(CWD_PATH, MODEL_NAME, 'frozen_inference_graph_LOSS_1,5.pb')

# Ruta al archivo que contiene etiquetas mapeadas a identificadores de objeto
# Este mapeo permite identificar con un nombre legible el valor predicho por 
#     la red convolutiva
PATH_TO_LABELS = os.path.join(CWD_PATH, 'labels', LABELS)


# Cargamos el mapeo de etiquetas.
# Para ello recurrimos a una libreria especializada, cargada previamente
label_map = label_map_util.load_labelmap(PATH_TO_LABELS)
categories = label_map_util.convert_label_map_to_categories(label_map, max_num_classes=NUM_CLASSES, use_display_name=True)
category_index = label_map_util.create_category_index(categories)

# Cargamos el modelo de TensorFlow en memoria
detection_graph = tf.compat.v1.Graph()
with detection_graph.as_default():
    od_graph_def = tf.compat.v1.GraphDef()
    with tf.gfile.GFile(PATH_TO_CKPT, 'rb') as fid:
        serialized_graph = fid.read()
        od_graph_def.ParseFromString(serialized_graph)
        tf.import_graph_def(od_graph_def, name='')

    TFSess = tf.compat.v1.Session(graph=detection_graph)

# Definimos los tensores de entrada y salida para el clasificador
# El tensor de entrada es cada cuadro del video (una imagen)
image_tensor = detection_graph.get_tensor_by_name('image_tensor:0')

# Los tensores de salida corresponden a las cajas de deteccion, scores, y clases
# Las cajas corresponden a la parte de la imagen que contiene un objeto detectado
detection_boxes = detection_graph.get_tensor_by_name('detection_boxes:0')

# Cada score representa el porcentaje de asertividad de la prediccion
# El score se muestra en conjunto con la etiqueta asignada al objeto detectado
detection_scores = detection_graph.get_tensor_by_name('detection_scores:0')
detection_classes = detection_graph.get_tensor_by_name('detection_classes:0')

# Numero de objetos detectados
num_detections = detection_graph.get_tensor_by_name('num_detections:0')

# Inicializar calculo de FPS (cuadros por segundo), para mostrarlo en pantalla
frame_rate_calc = 1
freq = cv2.getTickFrequency()
font = cv2.FONT_HERSHEY_SIMPLEX

# Nombre de la ventana del video a mostrar
WIN_NAME = 'Toñito\'s Detection'

# Inicializamos la camara
cap = cv2.VideoCapture(gstreamer_pipeline(flip_method=0), cv2.CAP_GSTREAMER)
if cap.isOpened():
    window_handle = cv2.namedWindow(WIN_NAME, cv2.WINDOW_AUTOSIZE)

frameCount = 0

while cv2.getWindowProperty(WIN_NAME,0) >= 0:

    t1 = cv2.getTickCount()

    # Obtenemos un cuadro del video, y expandimos sus dimensiones a la forma
    #   [1, None, None, 3], en concordancia con lo requerido por el tensor. Una sola 
    #   columna que contiene los valores RGB de cada pixel
    ret_val, frame = cap.read();
    frame.setflags(write=1)
    frame_expanded = np.expand_dims(frame, axis=0)#no optimizable

    # Realizamos la deteccion de objetos, proveyendo la imagen como entrada
    (boxes, scores, classes, num) = TFSess.run(
        [detection_boxes, detection_scores, detection_classes, num_detections],
        feed_dict={image_tensor: frame_expanded})

    #DEBUG
    #print(boxes)
    #print(np.squeeze(boxes))

    # Dibujamos los resultados de la deteccion sobre el video mostrado
    vis_util.visualize_boxes_and_labels_on_image_array(
        frame,
        np.atleast_2d(np.squeeze(boxes)),#no optimizable
        np.atleast_1d(np.squeeze(classes).astype(np.int32)),
        np.atleast_1d(np.squeeze(scores)),
        category_index,
        use_normalized_coordinates=True,
        line_thickness=8,
        #min_score_thresh=0.40)
        min_score_thresh=0.50)

    # Dibujamos los cuadros por segundo del video
    cv2.putText(frame,"FPS: {0:.2f}".format(frame_rate_calc),(30,50),font,1,(255,255,0),2,cv2.LINE_AA)

    # Mostramos la imagen con los dibujos superpuestos
    cv2.imshow(WIN_NAME, frame)

    # Calculo de FPS
    t2 = cv2.getTickCount()
    time1 = (t2-t1)/freq
    frame_rate_calc = 1/time1

    frameCount+=1
    #if frameCount == 3:
    #    break

    # Al presionar Q en el teclado, finalizamos la ejecucion
    if cv2.waitKey(1) == ord('q'):
        break

cap.release()

cv2.destroyAllWindows()

