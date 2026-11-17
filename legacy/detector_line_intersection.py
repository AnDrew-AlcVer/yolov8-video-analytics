import cv2
from ultralytics import YOLO

# ==========================================
# FUNCIÓN MATEMÁTICA PARA DETECTAR CRUCES
# ==========================================
# Esta función devuelve True si el segmento A-B cruza con el segmento C-D
def ccw(A, B, C):
    return (C[1] - A[1]) * (B[0] - A[0]) > (B[1] - A[1]) * (C[0] - A[0])

def intersectan(A, B, C, D):
    return ccw(A, C, D) != ccw(B, C, D) and ccw(A, B, C) != ccw(A, B, D)

print("Cargando el modelo YOLO...")
model = YOLO("yolov8s.pt") 

# IMPORTANTE: Cambia a otro vídeo donde sepas que alguien entra (ej. mov02.avi)
ruta_video = "mov07.avi" 
cap = cv2.VideoCapture(ruta_video)

if not cap.isOpened():
    print("Error: No se pudo abrir el vídeo.")
    exit()

# Tus coordenadas exactas
linea1_inicio = (280, 125)
linea1_fin = (385, 125)
linea2_inicio = (385, 125) 
linea2_fin = (415, 110)

# ==========================================
# VARIABLES DE MEMORIA Y CONTEO
# ==========================================
# Diccionario para guardar dónde estaba cada ID en el fotograma anterior
historial_posiciones = {} 

# Contadores de la puerta
entradas = 0
salidas = 0
# Lista para no contar a la misma persona dos veces
ids_contados = [] 

print("Reproduciendo vídeo. Pulsa 'q' para salir.")

while True:
    ret, frame = cap.read()
    if not ret:
        break

    resultados = model.track(frame, classes=[0], conf=0.4, persist=True)
    frame_anotado = resultados[0].plot()

    # Dibujamos las líneas de la puerta
    cv2.line(frame_anotado, linea1_inicio, linea1_fin, (255, 0, 0), 3)
    cv2.line(frame_anotado, linea2_inicio, linea2_fin, (255, 0, 0), 3)

    if resultados[0].boxes.id is not None:
        cajas = resultados[0].boxes.xyxy.cpu().numpy()
        ids = resultados[0].boxes.id.cpu().numpy().astype(int)

        for caja, id_persona in zip(cajas, ids):
            x1, y1, x2, y2 = caja
            cx = int((x1 + x2) / 2)
            cy = int(y2) 
            posicion_actual = (cx, cy)

            cv2.circle(frame_anotado, posicion_actual, 5, (0, 0, 255), -1)

            # LÓGICA DE CONTEO
            # Si ya tenemos registrada una posición anterior para este ID
            if id_persona in historial_posiciones:
                posicion_anterior = historial_posiciones[id_persona]
                
                # Solo comprobamos si no hemos contado ya a este ID
                if id_persona not in ids_contados:
                    # Comprobamos si el movimiento cruza la linea 1 o la linea 2
                    cruza_l1 = intersectan(posicion_anterior, posicion_actual, linea1_inicio, linea1_fin)
                    cruza_l2 = intersectan(posicion_anterior, posicion_actual, linea2_inicio, linea2_fin)

                    if cruza_l1 or cruza_l2:
                        # Si cruza de abajo hacia arriba (Y disminuye), está ENTRANDO al edificio
                        # Si cruza de arriba hacia abajo (Y aumenta), está SALIENDO a la plaza
                        if posicion_actual[1] < posicion_anterior[1]:
                            entradas += 1
                            print(f"ID {id_persona} ENTRÓ. Total Entradas: {entradas}")
                        else:
                            salidas += 1
                            print(f"ID {id_persona} SALIÓ. Total Salidas: {salidas}")
                        
                        # Añadimos el ID a la lista para no volver a contarlo
                        ids_contados.append(id_persona)

            # Actualizamos la memoria con la nueva posición
            historial_posiciones[id_persona] = posicion_actual

    # Mostrar los contadores en pantalla (Texto blanco, borde negro para que se lea bien)
    texto_entradas = f"Entradas a Carbonell: {entradas}"
    texto_salidas = f"Salidas a la Plaza: {salidas}"
    
    cv2.putText(frame_anotado, texto_entradas, (20, 40), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
    cv2.putText(frame_anotado, texto_salidas, (20, 80), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)

    cv2.imshow("Camara Universidad - Conteo", frame_anotado)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()