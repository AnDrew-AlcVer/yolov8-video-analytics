# YOLOv8 Pedestrian Zone Tracking & Occupancy Counter

Sistema de visión por computador para la detección, seguimiento multi-objeto (MOT) y conteo bidireccional de aforo (entradas, salidas y ocupación interior) en secuencias de videovigilancia exterior del edificio Carbonell (EPSA - UPV).

El sistema combina **YOLOv8** con el algoritmo de seguimiento **ByteTrack** y una máquina de estados personalizada por individuo que resuelve los problemas clásicos de oclusión, falsos cruces y oscilaciones en el umbral de puerta.

---

## Características Técnicas Principales

* **Doble Polígono de Control (ROI):**
  * **Zona de Detección Global (`ZONA_DETECCION`):** Filtra peatones irrelevantes que transitan por el fondo de la plaza, reduciendo falsos positivos y carga de seguimiento.
  * **Zona de Umbral de Puerta (`ZONA_PUERTA`):** Polígono de transición evaluado mediante `cv2.pointPolygonTest` sobre el punto de apoyo (centro inferior del *bounding box*).
* **Máquina de Estados por ID (`EstadoID`):**
  * **Debounce Asimétrico:** Requiere confirmación de múltiples fotogramas consecutivos (`FRAMES_DEBOUNCE_ENTRADA = 1`, `FRAMES_DEBOUNCE_SALIDA = 3`) para evitar conteos erróneos cuando una persona se detiene en el borde del polígono.
  * **Filtro de Paso Rápido (`MIN_FRAMES_EN_ZONA`):** Descarta y revierte automáticamente falsas entradas de peatones que rozan tangencialmente el umbral sin acceder al edificio.
  * **Gestión de Oclusiones y Memoria (`FRAMES_OLVIDO`):** Contabiliza entradas efectivas cuando un individuo procedente del exterior desaparece dentro del umbral de la puerta al cruzar hacia el interior.
* **Pipeline Opcional de Preprocesado:** Reducción de ruido de compresión (`fastNlMeansDenoisingColored`) y ecualización adaptativa de histograma (**CLAHE**) aplicada exclusivamente sobre el canal de luminancia ($L$) en el espacio de color CIELAB.
* **Modo de Calibración Visual:** Incluye la herramienta interactiva `calibrar()` para inspeccionar y ajustar los vértices numerados de ambos polígonos (`Dx` y `Px`) sobre el fotograma de referencia.

---

## Estructura del Repositorio

```text
yolov8-video-analytics/
├── detector.py          # Pipeline principal de inferencia, tracking, máquina de estados y calibración
├── requirements.txt     # Dependencias del entorno Python
├── .gitignore           # Exclusión de pesos (.pt) y vídeos brutos (.avi / .mp4)
└── README.md            # Documentación técnica del proyecto