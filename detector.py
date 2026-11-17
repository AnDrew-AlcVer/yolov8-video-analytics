# ============================================================
# DETECTOR DE PERSONAS — VERSIÓN LOCAL
# ============================================================
# Detecta y rastrea personas en un vídeo usando YOLOv8 +
# ByteTrack. Cuenta entradas y salidas a través de una zona
# de puerta definida por un polígono, y limita el área de
# detección a una zona personalizable.
# ============================================================

import cv2                      # Procesado de imagen y vídeo
import numpy as np              # Operaciones con arrays y polígonos
from ultralytics import YOLO    # Modelo de detección YOLOv8
import os                       # Manejo de rutas de ficheros
import torch


# ============================================================
# CONFIGURACIÓN CENTRAL
# ============================================================
# Todos los parámetros ajustables del sistema están aquí.
# Modifica solo esta sección para adaptar el detector a un
# nuevo vídeo o escenario.
# ============================================================
class Config:

    # ── Rutas ────────────────────────────────────────────────
    RUTA_BASE = os.path.dirname(os.path.abspath(__file__))  # Carpeta de trabajo
    NOMBRE_VIDEO_IN  = 'mov13.avi'          # Vídeo de entrada
    NOMBRE_VIDEO_OUT = 'video_resultado.mp4'  # Vídeo de salida con anotaciones

    # ── Modelo de detección ──────────────────────────────────
    # yolov8n (nano) es el más rápido, yolov8x (extra large)
    # el más preciso. El 'm' (medium) es un buen equilibrio.
    MODELO      = 'yolov8m.pt'
    CONF_UMBRAL = 0.20  # Confianza mínima para aceptar una detección (0–1)
    CLASES      = [0]   # Clase 0 = persona en el dataset COCO

    # Resolución de inferencia: mayor valor = más precisión a distancia
    # pero más consumo de VRAM. Valores típicos: 640, 960, 1280.
    IMGSZ   = 960

    # Test-Time Augmentation: hace varias pasadas con escalados y
    # volteos para mejorar detecciones difíciles. Más lento.
    AUGMENT = False

    # ── Preprocesado de imagen ───────────────────────────────
    # Activa CLAHE + denoising antes de la inferencia.
    # Útil para vídeos oscuros o con ruido de compresión.
    # Desactívalo si el vídeo tiene buena calidad (más rápido).
    PREPROCESAR        = False
    CLAHE_CLIP_LIMIT   = 2.0    # Límite de amplificación del contraste CLAHE
    CLAHE_TILE_GRID    = (8, 8) # Tamaño de la cuadrícula de tiles de CLAHE
    DENOISE_H          = 4      # Fuerza del denoising en luminancia
    DENOISE_H_COLOR    = 4      # Fuerza del denoising en color
    DENOISE_TEMPLATE_W = 7      # Tamaño del parche de comparación
    DENOISE_SEARCH_W   = 21     # Ventana de búsqueda del denoising

    # ── Zona de detección global ─────────────────────────────
    # Polígono que delimita el área donde se procesan personas.
    # Las detecciones fuera de este polígono se ignoran por
    # completo: no se rastrean, no se cuentan, no se dibujan.
    # Esto evita falsos conteos por personas que pasan al fondo
    # o en zonas irrelevantes de la escena.
    # Ajusta los vértices con la función calibrar().
    ZONA_DETECCION = np.array([
        [273,  70],   # vértice 0
        [362,  67],   # vértice 1
        [416,  55],   # vértice 2
        [413, 117],   # vértice 3
        [570, 200],   # vértice 4
        [640, 452],   # vértice 5
        [148, 430],   # vértice 6
        [  0, 350],   # vértice 7
        [  0, 125],   # vértice 8
        [280, 133],   # vértice 9
    ], dtype=np.int32)

    ZONA_DETECCION_COLOR = (255, 0, 0)  # Color del borde (BGR → azul)
    ZONA_DETECCION_ALPHA = 0.15         # Opacidad del relleno (0 = transparente)

    # ── Zona de puerta ───────────────────────────────────────
    # Polígono más pequeño que representa el umbral físico de
    # la puerta. Cuando el punto de pie de una persona cruza
    # este polígono (entra o sale), se registra la transición.
    # Debe estar contenido dentro de ZONA_DETECCION.
    ZONA_PUERTA = np.array([
        [273,  70],   # vértice 0
        [362,  67],   # vértice 1
        [416,  55],   # vértice 2
        [413, 117],   # vértice 3
        [383, 135],   # vértice 4
        [280, 133],   # vértice 5
    ], dtype=np.int32)

    ZONA_COLOR = (0, 200, 255)  # Color del borde (BGR → amarillo)
    ZONA_ALPHA = 0.25           # Opacidad del relleno

    # ── Debounce de transición ───────────────────────────────
    # Frames consecutivos en el nuevo estado necesarios para
    # confirmar una transición. Protege contra oscilaciones
    # cuando el punto de pie está justo en el borde del polígono.
    #
    # ENTRADA: valor bajo (1) porque el filtro de paso rápido
    #          ya descarta entradas falsas posteriores.
    # SALIDA:  valor más alto para evitar falsos conteos al
    #          salir brevemente de la zona.
    FRAMES_DEBOUNCE_ENTRADA = 1
    FRAMES_DEBOUNCE_SALIDA  = 3

    # ── Filtro de paso rápido ────────────────────────────────
    # Frames mínimos que una persona debe permanecer DENTRO de
    # ZONA_PUERTA para que su entrada (y posterior salida) se
    # cuente. Si sale antes, se cancela la entrada ya contada y
    # no se suma la salida. Sirve para filtrar personas que
    # rozan la zona sin realmente cruzar la puerta.
    # Referencia: a 25 fps, 25 frames ≈ 1 segundo.
    MIN_FRAMES_EN_ZONA = 20

    # ── Limpieza del historial ───────────────────────────────
    # Frames sin detectar un ID antes de eliminarlo del historial.
    # Si el tracker pierde a alguien y luego lo reencuentra con
    # un nuevo ID antes de este límite, el ID antiguo se borra.
    FRAMES_OLVIDO = 90

    # Frames iniciales del vídeo en los que no se cuentan las
    # primeras apariciones dentro de la zona (personas que ya
    # estaban en escena cuando empezó la grabación).
    FRAMES_GRACIA = 50


# ============================================================
# PREPROCESADO
# ============================================================

# Se instancia una sola vez aquí para no recrearlo en cada frame,
# lo que sería innecesariamente costoso en tiempo de CPU.
_CLAHE = cv2.createCLAHE(
    clipLimit=Config.CLAHE_CLIP_LIMIT,
    tileGridSize=Config.CLAHE_TILE_GRID,
)

def preprocesar_frame(frame: np.ndarray) -> np.ndarray:
    """
    Mejora la imagen antes de pasarla al modelo:
      1. Denoising: reduce el ruido de compresión del vídeo.
      2. CLAHE: mejora el contraste local en el canal de luminancia
         sin sobreexponer las zonas ya iluminadas.
    """
    # Paso 1: reducir ruido preservando los bordes
    frame = cv2.fastNlMeansDenoisingColored(
        frame, None,
        h=Config.DENOISE_H,
        hColor=Config.DENOISE_H_COLOR,
        templateWindowSize=Config.DENOISE_TEMPLATE_W,
        searchWindowSize=Config.DENOISE_SEARCH_W,
    )

    # Paso 2: CLAHE sobre el canal L del espacio LAB
    # (LAB separa luminancia de color, así CLAHE solo toca el brillo)
    lab = cv2.cvtColor(frame, cv2.COLOR_BGR2LAB)
    l, a, b = cv2.split(lab)
    lab = cv2.merge((_CLAHE.apply(l), a, b))
    return cv2.cvtColor(lab, cv2.COLOR_LAB2BGR)


# ============================================================
# DETECCIÓN DE ZONA
# ============================================================

def punto_en_zona(punto: tuple, zona: np.ndarray) -> bool:
    """
    Devuelve True si el punto (x, y) está dentro o sobre el borde
    del polígono dado. Usa cv2.pointPolygonTest, que devuelve:
      +distancia  → dentro
       0          → sobre el borde
      -distancia  → fuera
    Con measureDist=False devuelve +1, -1 o 0, que es más rápido.
    """
    resultado = cv2.pointPolygonTest(
        zona, (float(punto[0]), float(punto[1])), False
    )
    return resultado >= 0


# ============================================================
# ESTADO POR ID
# ============================================================

class EstadoID:
    """
    Gestiona el ciclo de vida de un ID de tracking respecto a
    la zona de puerta. Registra si la persona está dentro o fuera,
    aplica debounce para evitar conteos falsos en el borde, y
    filtra los pasos demasiado rápidos.

    Atributos:
      estado            — estado confirmado actual: 'dentro' o 'fuera'
      estado_candidato  — nuevo estado en observación (pendiente de debounce)
      frames_candidato  — frames consecutivos en el estado candidato
      ultimo_frame      — último frame en que se detectó este ID
      frame_entro       — frame en que se confirmó la última entrada
      entrada_contada   — True si la entrada actual ya se sumó al contador
      visto_fuera       — True si este ID alguna vez estuvo fuera de la zona
                          (protege contra contar como entrada a alguien que
                          en realidad estaba saliendo del edificio)
    """
    __slots__ = (
        'estado', 'estado_candidato', 'frames_candidato',
        'ultimo_frame', 'frame_entro', 'entrada_contada',
        'visto_fuera',
    )

    def __init__(self, dentro_inicial: bool, frame_actual: int):
        self.estado           = 'dentro' if dentro_inicial else 'fuera'
        self.estado_candidato = self.estado
        self.frames_candidato = 0
        self.ultimo_frame     = frame_actual
        self.frame_entro      = frame_actual if dentro_inicial else None
        self.entrada_contada  = False
        # Si aparece por primera vez fuera, marcamos que ya lo hemos
        # visto fuera; si aparece dentro, aún no sabemos su origen.
        self.visto_fuera      = not dentro_inicial

    def actualizar(self, dentro: bool, frame_actual: int,
                   debounce_entrada: int, debounce_salida: int,
                   min_frames_zona: int):
        """
        Actualiza el estado del ID con el nuevo dato de posición.

        Retorna:
          'entrada'     — transición fuera→dentro confirmada y válida
          'salida'      — transición dentro→fuera confirmada y válida
          'paso_rapido' — salida confirmada pero la persona estuvo
                          menos de min_frames_zona dentro; la entrada
                          previa se debe cancelar
          None          — sin transición relevante este frame
        """
        self.ultimo_frame = frame_actual
        nuevo = 'dentro' if dentro else 'fuera'

        # Registramos si en algún momento se detectó fuera de la zona
        if nuevo == 'fuera':
            self.visto_fuera = True

        # Si el nuevo estado coincide con el estado estable actual,
        # no hay transición: reiniciamos el contador de candidato.
        if nuevo == self.estado:
            self.estado_candidato = self.estado
            self.frames_candidato = 0
            return None

        # Acumulamos frames en el estado candidato o lo reiniciamos
        # si el candidato cambió (por ejemplo, osciló al borde).
        if nuevo == self.estado_candidato:
            self.frames_candidato += 1
        else:
            self.estado_candidato = nuevo
            self.frames_candidato = 1

        # Debounce asimétrico: menos estricto para entradas (ya las
        # filtra el paso_rapido) y más estricto para salidas.
        debounce = debounce_entrada if nuevo == 'dentro' else debounce_salida
        if self.frames_candidato < debounce:
            return None  # Aún no confirmado, seguir observando

        # ── Transición confirmada ──────────────────────────────
        self.estado           = nuevo
        self.estado_candidato = nuevo
        self.frames_candidato = 0

        if nuevo == 'dentro':
            # Registramos el frame de entrada para medir cuánto
            # tiempo pasa la persona dentro de la zona.
            self.frame_entro     = frame_actual
            self.entrada_contada = True
            return 'entrada'

        else:  # nuevo == 'fuera': la persona acaba de salir
            tiempo_dentro = frame_actual - (self.frame_entro or frame_actual)

            # Filtro de paso rápido: si la persona estuvo muy poco
            # tiempo dentro, consideramos que no fue una entrada real
            # (p.ej. alguien que rozó la zona sin cruzar la puerta).
            if self.entrada_contada and tiempo_dentro < min_frames_zona:
                self.entrada_contada = False
                return 'paso_rapido'  # Señal para cancelar la entrada

            self.entrada_contada = False
            return 'salida'


# ============================================================
# DIBUJADO
# ============================================================

def dibujar_zona(frame, zona, color, alpha):
    """
    Dibuja un polígono con relleno semitransparente y borde sólido.
    Usa una copia del frame (overlay) para aplicar la transparencia
    con cv2.addWeighted sin modificar el original.
    """
    overlay = frame.copy()
    cv2.fillPoly(overlay, [zona], color)
    cv2.addWeighted(overlay, alpha, frame, 1 - alpha, 0, frame)
    cv2.polylines(frame, [zona], isClosed=True, color=color, thickness=2)
    return frame


def dibujar_punto_pie(frame, punto, dentro):
    """
    Dibuja un punto en los pies de cada persona detectada.
    Verde si está dentro de la zona de puerta, rojo si está fuera.
    Ayuda a verificar visualmente si la lógica de cruce funciona bien.
    """
    color = (0, 255, 80) if dentro else (0, 100, 255)
    cv2.circle(frame, punto, 4, color, -1)
    return frame


def dibujar_panel(frame, entradas, salidas, interior, detectados_ahora):
    """
    Dibuja el panel de estadísticas en la esquina superior izquierda.
    Usa un rectángulo negro semitransparente como fondo para que el
    texto sea legible sobre cualquier color de imagen.

    Contadores mostrados:
      - Detectados ahora : personas en ZONA_DETECCION en este frame
      - Entradas         : total acumulado de entradas al edificio
      - Salidas          : total acumulado de salidas del edificio
      - Interior (E-S)   : estimación de personas dentro del edificio
    """
    # Fondo semitransparente
    overlay = frame.copy()
    cv2.rectangle(overlay, (20, 20), (320, 150), (0, 0, 0), -1)
    cv2.addWeighted(overlay, 0.5, frame, 0.5, 0, frame)
    cv2.rectangle(frame, (20, 20), (320, 150), (200, 200, 200), 1)

    # Textos con colores diferenciados por tipo de dato
    cv2.putText(frame, f"Detectados ahora: {detectados_ahora}", (35, 52),
                cv2.FONT_HERSHEY_SIMPLEX, 0.65, (255, 255, 255), 2)
    cv2.putText(frame, f"Entradas:         {entradas}",         (35, 80),
                cv2.FONT_HERSHEY_SIMPLEX, 0.65, (150, 255, 150), 2)
    cv2.putText(frame, f"Salidas:          {salidas}",          (35, 108),
                cv2.FONT_HERSHEY_SIMPLEX, 0.65, (150, 150, 255), 2)
    cv2.putText(frame, f"Interior (E-S):   {interior}",         (35, 136),
                cv2.FONT_HERSHEY_SIMPLEX, 0.65, (255, 200, 100), 2)
    return frame


# ============================================================
# BUCLE PRINCIPAL
# ============================================================

def main():
    ruta_in  = os.path.join(Config.RUTA_BASE, Config.NOMBRE_VIDEO_IN)
    ruta_out = os.path.join(Config.RUTA_BASE, Config.NOMBRE_VIDEO_OUT)

    # Abrir el vídeo de entrada
    cap = cv2.VideoCapture(ruta_in)
    if not cap.isOpened():
        raise RuntimeError(f"No se pudo abrir el vídeo: {ruta_in}")

    # Leer propiedades del vídeo para configurar la salida con los
    # mismos parámetros (resolución y framerate)
    w   = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    h   = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = int(cap.get(cv2.CAP_PROP_FPS))

    # Escritor del vídeo de salida en formato MP4
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out    = cv2.VideoWriter(ruta_out, fourcc, fps, (w, h))

    # Cargar el modelo y moverlo a la GPU
    print("\n🚀 Cargando modelo...")
    modelo_yolo = YOLO(Config.MODELO)
    dispositivo = 'cuda' if torch.cuda.is_available() else 'cpu'
    modelo_yolo.to(dispositivo)  # Usar GPU NVIDIA; cambiar a 'cpu' si no hay GPU
    print(f"Dispositivo en uso: {modelo_yolo.device}")
    modelo_yolo.names[0] = ""  # Oculta la etiqueta "person" en los bounding boxes

    # Estado global del sistema
    entradas_totales = 0   # Acumulador de entradas al edificio
    salidas_totales  = 0   # Acumulador de salidas del edificio
    estados: dict[int, EstadoID] = {}  # Historial de estado por ID de tracking
    frame_count = 0        # Contador de frames procesados

    print(f"--- Procesando: {Config.NOMBRE_VIDEO_IN} ---")
    print(f"    imgsz={Config.IMGSZ} | augment={Config.AUGMENT} | preprocesar={Config.PREPROCESAR}")
    print("\nPresiona Q para detener el procesado.\n")

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break  # Fin del vídeo o error de lectura

        frame_count += 1

        # Preprocesar el frame si está activado (mejora para vídeos oscuros)
        frame_inferencia = preprocesar_frame(frame) if Config.PREPROCESAR else frame

        # ── Inferencia + tracking ──────────────────────────────
        # model.track() ejecuta detección Y tracking en un solo paso.
        # persist=True mantiene los IDs entre frames (necesario para ByteTrack).
        resultados = modelo_yolo.track(
            frame_inferencia,
            classes=Config.CLASES,
            conf=Config.CONF_UMBRAL,
            imgsz=Config.IMGSZ,
            augment=Config.AUGMENT,
            tracker="bytetrack.yaml",
            persist=True,
            verbose=False,
        )

        # Dibujar bounding boxes (lo hace ultralytics automáticamente)
        frame_out = resultados[0].plot(line_width=1, conf=False)

        # Dibujar las zonas encima de los bounding boxes
        dibujar_zona(frame_out, Config.ZONA_DETECCION,
                     Config.ZONA_DETECCION_COLOR, Config.ZONA_DETECCION_ALPHA)
        dibujar_zona(frame_out, Config.ZONA_PUERTA,
                     Config.ZONA_COLOR, Config.ZONA_ALPHA)

        personas_detectadas_ahora = 0  # Personas visibles en este frame

        if resultados[0].boxes.id is not None:
            cajas = resultados[0].boxes.xyxy.cpu().numpy()   # Coordenadas [x1,y1,x2,y2]
            ids   = resultados[0].boxes.id.cpu().numpy().astype(int)  # IDs del tracker
            ids_vistos = set()  # IDs activos en este frame (para la limpieza)

            for caja, id_obj in zip(cajas, ids):
                x1, y1, x2, y2 = caja
                # El punto de pie (centro inferior del bounding box) es
                # más estable que el centro para detectar cruces de zona,
                # ya que representa dónde toca el suelo la persona.
                pie = (int((x1 + x2) / 2), int(y2))

                # Ignorar detecciones fuera de la zona de detección global
                if not punto_en_zona(pie, Config.ZONA_DETECCION):
                    continue

                personas_detectadas_ahora += 1
                ids_vistos.add(id_obj)

                # Comprobar si el pie está dentro de la zona de puerta
                dentro_puerta = punto_en_zona(pie, Config.ZONA_PUERTA)

                if id_obj not in estados:
                    # Primera vez que vemos este ID: inicializar su estado.
                    # No contamos nada aquí; esperamos a observar su movimiento.
                    estados[id_obj] = EstadoID(dentro_puerta, frame_count)
                else:
                    # Actualizar el estado y procesar la posible transición
                    resultado = estados[id_obj].actualizar(
                        dentro_puerta, frame_count,
                        Config.FRAMES_DEBOUNCE_ENTRADA,
                        Config.FRAMES_DEBOUNCE_SALIDA,
                        Config.MIN_FRAMES_EN_ZONA,
                    )
                    if resultado == 'entrada':
                        entradas_totales += 1
                    elif resultado == 'salida':
                        salidas_totales += 1
                    elif resultado == 'paso_rapido':
                        # La entrada ya fue sumada, pero el tiempo dentro
                        # fue insuficiente: la deshacemos.
                        entradas_totales -= 1

                dibujar_punto_pie(frame_out, pie, dentro_puerta)

            # ── Limpieza de IDs ausentes ───────────────────────
            # Eliminamos del historial los IDs que llevan más de
            # FRAMES_OLVIDO frames sin aparecer en el vídeo.
            ids_a_borrar = [
                k for k, e in estados.items()
                if frame_count - e.ultimo_frame > Config.FRAMES_OLVIDO
                   and k not in ids_vistos
            ]
            for k in ids_a_borrar:
                e = estados[k]
                # Caso especial: el ID desapareció mientras estaba dentro
                # de la zona y nunca fue contado como entrada, pero sí
                # estuvo fuera antes → entró al edificio sin que el
                # debounce lo alcanzara a confirmar. Lo contamos ahora.
                if (e.estado == 'dentro'
                        and not e.entrada_contada
                        and e.visto_fuera
                        and frame_count > Config.FRAMES_GRACIA):
                    entradas_totales += 1
                del estados[k]

        # Personas estimadas dentro del edificio en este momento
        interior = entradas_totales - salidas_totales

        dibujar_panel(frame_out, entradas_totales, salidas_totales,
                      interior, personas_detectadas_ahora)
        out.write(frame_out)

        # Mostrar el frame procesado en tiempo real
        cv2.imshow('Detector', frame_out)
        # Esperar 1ms; si se pulsa Q, interrumpir el procesado
        if cv2.waitKey(1) & 0xFF == ord('q'):
            print("\nProcesado interrumpido por el usuario.")
            break

    # Liberar recursos
    cap.release()
    out.release()
    cv2.destroyAllWindows()

    print(f"\n✅ Vídeo guardado en: {ruta_out}")
    print(f"   Entradas totales  : {entradas_totales}")
    print(f"   Salidas totales   : {salidas_totales}")
    print(f"   Interior final    : {entradas_totales - salidas_totales}")


# ============================================================
# CALIBRACIÓN
# ============================================================
# Función auxiliar para ajustar visualmente las zonas antes
# de procesar el vídeo completo. Muestra el primer frame con
# ambas zonas superpuestas y los vértices numerados.
#
# Uso: comenta main() al final del archivo y descomenta calibrar()
# ============================================================

def calibrar():
    ruta_in = os.path.join(Config.RUTA_BASE, Config.NOMBRE_VIDEO_IN)
    cap = cv2.VideoCapture(ruta_in)
    ret, frame = cap.read()
    cap.release()
    if not ret:
        raise RuntimeError(f"No se pudo leer el primer frame de: {ruta_in}")

    # Dibujar la zona de detección global (vértices en azul, etiqueta Dx)
    dibujar_zona(frame, Config.ZONA_DETECCION, Config.ZONA_DETECCION_COLOR, 0.2)
    for i, (vx, vy) in enumerate(Config.ZONA_DETECCION):
        cv2.circle(frame, (vx, vy), 5, (255, 100, 100), -1)
        cv2.putText(frame, f"D{i}", (vx + 6, vy - 6),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.55, (255, 100, 100), 2)

    # Dibujar la zona de puerta encima (vértices en blanco, etiqueta Px)
    dibujar_zona(frame, Config.ZONA_PUERTA, Config.ZONA_COLOR, 0.35)
    for i, (vx, vy) in enumerate(Config.ZONA_PUERTA):
        cv2.circle(frame, (vx, vy), 5, (255, 255, 255), -1)
        cv2.putText(frame, f"P{i}", (vx + 6, vy - 6),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.55, (255, 255, 255), 2)

    # Imprimir coordenadas actuales en la terminal para referencia
    print("Zona de DETECCIÓN GLOBAL (puntos azules 'Dx'):")
    for i, v in enumerate(Config.ZONA_DETECCION):
        print(f"  D{i}: {tuple(v)}")
    print("\nZona de PUERTA (puntos blancos 'Px'):")
    for i, v in enumerate(Config.ZONA_PUERTA):
        print(f"  P{i}: {tuple(v)}")
    print("\nPulsa cualquier tecla para cerrar la ventana.")

    cv2.imshow('Calibración  (Dx = Detección | Px = Puerta)', frame)
    cv2.waitKey(0)
    cv2.destroyAllWindows()


# ============================================================
# BUCLE PRINCIPAL O CALIBRACIÓN (Descomentar la función a ejecutar)
# ============================================================

if __name__ == '__main__':
    main()
    # calibrar()