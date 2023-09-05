import cv2

# Leemos los pesos y la arquitectura
model = 'pose_deploy_linevec_faster_4_stages.prototxt'
pesos = 'pose_iter_160000.caffemodel'

# Definimos el número de puntos y sus uniones
numPuntos = 15
pares = [[0,1], [1,2], [2,3], [3,4], [1,5], [5,6], [6,7], [1,14], [14,8], [8,9], [9,10], [14,11], [11,12], [12,13]]

# Leemos nuestros pesos y arquitectura
net = cv2.dnn.readNetFromCaffe(model, pesos)

# Creamos la video captura
cap = cv2.VideoCapture(0)

# Inicializamos variables
p = False
e = False

# Creamos un ciclo para ejeecutar nuestros frames
while True:
    # Leemos los fotogramas
    ret, frame = cap.read()

    # Corregimos color
    rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

    # Extraemos ancho y alto
    anchoFrame = frame.shape[1]
    altoFrame = frame.shape[0]

    # Preprocesamos nuestros frames
    # Es decir convertimos nuestros frames a escenarios que sean parecidos a los que fue entrenado el modelo
    TamEntNet = (368,368)
    # valores = rgb, valor de escala, tamaño, formato de color, formato de canal y no haga recortes en imagen
    blob = cv2.dnn.blobFromImage(rgb, 1.0 / 255, TamEntNet, (0,0,0), swapRB= True, crop= False)

    # Entregamos las imagenes procesadas a la CNN = Red neuronal
    net.setInput(blob)

    # Extraemos info o resultados de la red neuronal
    output = net.forward()

    # Escalamos la salida al tamaño de nuestros frames
    scalex = anchoFrame / output.shape[3]
    scaley = altoFrame / output.shape[2]

    # mostramos el mapa de probabilidad de los resultados
    for i in range(numPuntos):
        # Extraemos el mapa
        map = output[0, i, :, :]
        # Redimensionamos
        dismap = cv2.resize(map, (anchoFrame, altoFrame), cv2.INTER_LINEAR)
        cv2.imshow('MAPA', dismap)


    # Mostramos los frames
    cv2.imshow("VIDEOCAPTURA", frame)

    # Cerramos con lectura de teclado
    t = cv2.waitKey(1)
    if t == 27:
        break

    # Si queremos dibujar los puntos
    if t == 122 or t == 80:
        p == True
        e == False

    # Si queremos dibujar los puntos
    if t == 101 or t == 69:
        p = False
        e = True

# Liberamos la VideoCaptura
cap.release()
# Cerramos la ventana
cv2.destroyAllWindows()