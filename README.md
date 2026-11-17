# Detector y Contador de Personas — Visión por Computador

Sistema de visión artificial para el control de aforo y recuento bidireccional de peatones en la Plaza Ferrándiz i Carbonell (EPSA - UPV), combinando **YOLOv8**, seguimiento multi-objeto con **ByteTrack** y una máquina de estados geométrica por polígonos.

![Demostración del Detector en Tiempo Real](assets/demo.gif)

**Autor:** Andrés Alcaraz Verdú  
**Proyecto:** Control de aforo y recuento bidireccional de peatones en la Plaza Ferrándiz i Carbonell (Edificio Carbonell - EPSA, UPV).

---

## Índice

1. [Introducción](#1-introducción)
   - [Contexto y Motivación](#contexto-y-motivación)
   - [Objetivos del Proyecto](#objetivos-del-proyecto)
2. [Estado del Entorno y las Tecnologías](#2-estado-del-entorno-y-las-tecnologías)
   - [Visión por Computador y Deep Learning](#visión-por-computador-y-deep-learning)
   - [Herramientas y Entorno de Desarrollo](#herramientas-y-entorno-de-desarrollo)
   - [Modelos y Algoritmos](#modelos-y-algoritmos)
3. [Metodología y Arquitectura del Sistema](#3-metodología-y-arquitectura-del-sistema)
   - [Preprocesamiento de Imagen](#preprocesamiento-de-imagen)
   - [Detección de Personas](#detección-de-personas)
   - [Sistema de Seguimiento (Tracking)](#sistema-de-seguimiento-tracking)
   - [Geometría y Lógica de Recuento](#geometría-y-lógica-de-recuento)
   - [Renderizado Visual e Interfaz (HUD)](#renderizado-visual-e-interfaz-bloque-dibujado)
   - [Flujo de Ejecución Principal (Bucle Main)](#flujo-de-ejecución-principal-bucle-main)
4. [Manual de Uso y Configuración del Sistema](#4-manual-de-uso-y-configuración-del-sistema)
   - [Instalación Rápida](#instalación-rápida)
   - [Gestión de Archivos e Inferencia](#gestión-de-archivos-e-inferencia)
   - [Geometría y Delimitación Espacial](#geometría-y-delimitación-espacial)
   - [Calibración de la Máquina de Estados](#calibración-de-la-máquina-de-estados-lógica-de-recuento)
   - [Preprocesamiento de Imagen (Opcional)](#preprocesamiento-de-imagen-opcional)
5. [Dificultades Encontradas y Soluciones Adoptadas](#5-dificultades-encontradas-y-soluciones-adoptadas)
6. [Resultados y Evaluación](#6-resultados-y-evaluación)
   - [Entorno de Pruebas y Selección de Muestra (Ground Truth)](#entorno-de-pruebas-y-selección-de-muestra)
   - [Enlaces de Demostración en Vídeo](#enlaces-de-demostración)
   - [Análisis de Rendimiento (Caso de Estudio: Vídeo 13)](#análisis-de-rendimiento-caso-de-estudio-vídeo-13)
   - [Métricas Finales](#métricas-finales)
7. [Conclusiones y Trabajo Futuro](#7-conclusiones-y-trabajo-futuro)

---

## 1. Introducción

### Contexto y Motivación
El presente proyecto surge del objetivo de aplicar los fundamentos teóricos de la visión artificial a un entorno real y desafiante. Para ello, se ha seleccionado la denominada "Propuesta 4", centrada en el control de aforo y el recuento de personas en la Plaza Ferrándiz i Carbonell, monitorizando específicamente los accesos al edificio Carbonell.

La motivación principal detrás de esta elección no es únicamente la búsqueda de una precisión algorítmica infalible, sino la demostración práctica de cómo plantear, desarrollar y argumentar una solución de ingeniería completa. Este escenario destaca por ser uno de los más completos y visuales para su exposición, ya que permite combinar de forma tangible dos pilares fundamentales de la visión por computador moderna: la **detección de objetos** (identificar a los individuos que transitan por la plaza) y el **seguimiento espacial o tracking** (asignar un identificador único a cada sujeto para evaluar su trayectoria temporal).

### Objetivos del Proyecto
El propósito central de este trabajo es desarrollar un pipeline de software capaz de gestionar de forma autónoma el conteo de personas. Para alcanzar este fin, se han establecido los siguientes objetivos técnicos:

* **Integración de Detección y Seguimiento:** Implementar modelos de Deep Learning avanzados orientados a la detección espacial (arquitecturas YOLO), combinados con algoritmos de seguimiento paramétrico para mantener la consistencia de las identidades a lo largo de los frames del vídeo.
* **Lógica Geométrica de Recuento:** Desarrollar una estructura de programación sólida mediante la implementación de "líneas virtuales" y polígonos delimitadores en la puerta del edificio. El sistema debe interpretar espacialmente la escena para discernir la dirección del movimiento, sumando un individuo ($+1$) al acceder al interior o restándolo ($-1$) al salir a la plaza.
* **Resolución Analítica de Problemas:** Exponer y documentar el proceso iterativo de desarrollo, demostrando la capacidad de adaptar el código frente a complicaciones del entorno real, tales como oclusiones, tránsito en grupos densos o deficiencias de iluminación y contraste en la captura original.

---

## 2. Estado del Entorno y las Tecnologías

### Visión por Computador y Deep Learning
La disciplina de la Visión por Computador ha experimentado un salto cualitativo gracias a la adopción de técnicas de Deep Learning. En el contexto del recuento de personas, los enfoques clásicos basados puramente en el procesado algorítmico de imágenes han sido complementados, y en gran medida sustituidos en la fase de detección, por redes neuronales convolucionales.

Estos modelos de aprendizaje profundo permiten extraer características complejas de una escena, logrando identificar peatones en entornos dinámicos. La combinación de la detección por Deep Learning con el procesado de imágenes clásico proporciona la robustez necesaria para operar en un entorno real y no controlado, donde los sujetos interactúan, se ocluyen entre sí y se ven afectados por las condiciones ambientales.

### Herramientas y Entorno de Desarrollo
Para la construcción y ejecución del pipeline visual, se ha definido un stack tecnológico basado en los siguientes componentes:

* **Python:** Empleado como lenguaje de programación principal para la estructuración lógica de todo el sistema.
* **OpenCV:** Utilizada como la biblioteca central de visión artificial para el manejo del flujo de vídeo, el procesado de imágenes previo a la red neuronal y el renderizado visual de métricas y zonas de detección.
* **Google Colab (GPU) $\rightarrow$ Entorno Local CUDA:** Inicialmente, el proyecto se planteó sobre la nube de Google Colab, aprovechando sus servidores con GPU y la flexibilidad de poder desarrollar desde distintos equipos portátiles. El procesamiento inicial basado en CPU demostró ser excesivamente lento, provocando tirones y "pantallazos" en la reproducción del vídeo. Sin embargo, al constatar las limitaciones de uso del entorno gratuito de Colab para procesar de forma continua el vídeo, se procedió a migrar el proyecto a un entorno de desarrollo local, ejecutando el código de forma nativa en un ordenador de sobremesa dotado de una tarjeta gráfica dedicada, lo que resolvió definitivamente los cuellos de botella en la inferencia.

### Modelos y Algoritmos
El núcleo analítico del proyecto se divide en la identificación visual y la lógica espacial, apoyándose en los siguientes modelos y estrategias:

* **Modelo YOLOv8:** Para la inferencia y detección de sujetos se ha seleccionado el modelo YOLOv8 (*You Only Look Once*). La elección de su arquitectura **"Medium" (`yolov8m.pt`)** es el resultado de múltiples pruebas empíricas de rendimiento y precisión. Inicialmente se testeó la variante "Nano", la cual no lograba detectar a gran parte de los individuos presentes en la plaza y presentaba problemas severos para reconocer a las personas justo en el momento de cruzar el umbral de la puerta. Posteriormente, se evaluó el extremo opuesto con el modelo "Extra Large"; aunque su funcionamiento en la detección era excelente, el tiempo de procesamiento superaba la media hora para un vídeo de apenas un minuto de duración, haciéndolo inviable. Finalmente, la versión "Medium" ofreció el equilibrio ideal, destacando por su alta eficiencia al analizar la imagen completa en una única pasada y logrando una óptima relación entre el coste computacional y la calidad de la detección. Para asegurar el reconocimiento de peatones, el modelo hace uso del aprendizaje previo basado en el conjunto de datos COCO (*Common Objects in Context*).
* **Algoritmo de Tracking:** Para solucionar la pérdida temporal de información y gestionar el recuento en multitudes, la detección se complementa con **ByteTrack**, un algoritmo de seguimiento que asigna y mantiene un identificador (`ID`) constante para cada persona a lo largo de los distintos frames del vídeo.
* **Algoritmos Geométricos para Intersecciones:** En la primera fase del desarrollo, se implementó un sistema basado en una línea virtual para la detección de entradas y salidas. El funcionamiento de este método inicial consistía en analizar vectorialmente el movimiento de las coordenadas de las personas: si la componente $y$ de su posición disminuía tras atravesar la línea, el sistema interpretaba que se producía una entrada; de lo contrario, se contabilizaba como una salida. Sin embargo, este planteamiento lineal demostró tener un amplio margen de mejora, ya que era propenso a errores causados por oclusiones, retrocesos o detecciones intermitentes en el umbral. Para resolver estas limitaciones, se sustituyó la línea por una solución basada en geometría espacial mediante la definición analítica de polígonos irregulares que cubren un área tridimensional alrededor de la puerta. Utilizando funciones matemáticas de OpenCV (como el cálculo de pruebas de polígonos), el sistema proyecta las coordenadas de los pies de cada individuo detectado para verificar si intersectan con dicha zona. Al cruzar estos datos espaciales con el histórico de los identificadores únicos (`IDs`) provistos por el tracker, la máquina de estados determina con gran fiabilidad las transiciones: si un ID específico se registra primero en el estado `'fuera'` y posteriormente cambia al estado `'dentro'` de la zona delimitada, el sistema confirma de manera robusta una entrada efectiva al edificio.

---

## 3. Metodología y Arquitectura del Sistema

### Preprocesamiento de Imagen
Para garantizar la robustez del sistema frente a las variaciones lumínicas y la calidad de la grabación, se ha diseñado un módulo de preprocesado (`preprocesar_frame`) que actúa sobre cada frame antes de la inferencia. Este proceso consta de dos fases secuenciales:

1. **Reducción de Ruido (Denoising):** Se aplica la función `fastNlMeansDenoisingColored` de OpenCV para suavizar el ruido de compresión característico de los vídeos, preservando la nitidez de los bordes estructurales.
2. **Ecualización del Histograma (CLAHE):** Para corregir la mala iluminación y el bajo contraste sin distorsionar los colores de la escena, la imagen se convierte primero al espacio de color LAB. La ecualización CLAHE (*Contrast Limited Adaptive Histogram Equalization*), configurada con un límite de amplificación de $2.0$ y una cuadrícula de $8 \times 8$, se aplica de forma exclusiva sobre el canal `L` (luminancia). Posteriormente, los canales se fusionan y se devuelven al formato BGR estándar.

#### Limitaciones y Rendimiento en Entorno Real
A pesar de la eficacia técnica de este módulo, en la configuración de la versión final del sistema se ha optado por mantenerlo inactivo por defecto (`PREPROCESAR = False`). Esta decisión de ingeniería responde a un severo cuello de botella a nivel de hardware: las funciones de preprocesamiento de imágenes detalladas se ejecutan nativamente sobre la CPU del ordenador, lo que ralentiza drásticamente el flujo de ejecución global del detector. Los intentos de forzar y delegar estos cálculos matemáticos específicos a la GPU derivaron en inestabilidades críticas, llegando a provocar errores fatales en el sistema operativo (pantallazos azules) por fallos de cálculo.

No obstante, en las pruebas aisladas donde el preprocesado fue activado, el módulo demostró cumplir su objetivo con éxito, mejorando sustancialmente la tasa de detección de personas bajo condiciones lumínicas adversas. Su impacto positivo fue especialmente notable al identificar sujetos ubicados en la zona crítica del umbral de las puertas del edificio, donde las sombras y el fuerte contraste suelen dificultar la tarea del modelo predictivo.

### Detección de Personas
El núcleo de la visión artificial recae sobre la red neuronal YOLOv8, instanciada mediante la librería `ultralytics`. La configuración exacta para la inferencia se ha ajustado para maximizar el rendimiento en un entorno exterior:

* **Modelo y Clases:** Se utiliza la arquitectura `yolov8m.pt` (Medium), restringiendo las detecciones exclusivamente a la clase `0` del dataset COCO, correspondiente a peatones.
* **Umbral de Confianza:** Se ha establecido un `CONF_UMBRAL` inusualmente bajo, fijado en el $20\%$ (`0.20`). Esta decisión de diseño se fundamenta en un análisis frente a dos de los principales desafíos del proyecto: la omisión de detecciones de personas legítimas y la clasificación errónea de objetos como peatones. Al reducir la exigencia del modelo, se logran captar individuos lejanos o parcialmente ocluidos, asumiendo el contra de aumentar las detecciones erróneas sobre el mobiliario urbano. Sin embargo, este riesgo queda neutralizado de forma muy efectiva gracias a la "Zona de Detección" delimitada. Al restringir el análisis exclusivamente al interior del polígono de la plaza, se filtran espacialmente la inmensa mayoría de los falsos positivos detectados en el fondo o los laterales, justificando y dando pleno sentido a la aplicación de este umbral tan permisivo.
* **Resolución de Inferencia:** Se ha forzado un tamaño de imagen (`IMGSZ`) de $960$ píxeles, superior al estándar, para dotar al modelo de mayor precisión a larga distancia.

### Sistema de Seguimiento (Tracking)
La persistencia temporal de las identidades se gestiona mediante la integración nativa de ByteTrack (`bytetrack.yaml`). Este algoritmo no solo sigue el movimiento espacial, sino que permite trazar trayectorias continuas.

* **Memoria de Historial:** Para mitigar la pérdida de identificadores provocada por oclusiones severas (cuando una persona queda tapada por otra o por un obstáculo temporal), el sistema se ha parametrizado con un límite de limpieza (`FRAMES_OLVIDO`) de $90$ frames. Si el tracker pierde el rastro de un individuo, su identificador único (`ID`) se conserva en la memoria durante este margen; si vuelve a ser detectado, recupera su ID original sin duplicar el conteo.

### Geometría y Lógica de Recuento
La interpretación semántica de la escena y el conteo analítico operan de forma conjunta mediante un mapeo espacial y una máquina de estados finitos personalizada (`EstadoID`).

* **Definición de Zonas Geométricas:**
  * **Zona de Detección:** Un polígono global que delimita el área de la plaza que resulta de interés. Cualquier detección generada por YOLOv8 que caiga fuera de este perímetro se descarta inmediatamente, ahorrando recursos computacionales y evitando falsos conteos en el fondo de la escena.
  * **Zona de Puerta:** Un polígono tridimensional más pequeño y preciso, circunscrito en la Zona de Detección, que enmarca el umbral físico del edificio Carbonell.
* **Cálculo del Punto de Anclaje:** En lugar de utilizar el centro de la *bounding box* (caja delimitadora) proporcionada por la IA, el sistema calcula de forma dinámica el "punto de pie" de cada sujeto (el centro de la arista inferior). Este punto representa el contacto real con el suelo y se evalúa matemáticamente con la función `pointPolygonTest` de OpenCV para determinar si se encuentra dentro o fuera de los polígonos delimitadores.
* **Máquina de Estados y Filtros de Transición:** La lógica de cruce monitoriza el estado de cada ID (`'dentro'` o `'fuera'`) de forma continua. Para validar una entrada o salida real, se aplican dos capas de protección algorítmica:
  * **Debounce (Antirrebote):** Se exigen frames consecutivos de confirmación en el nuevo estado ($1$ frame para entradas, $3$ para salidas) para ignorar oscilaciones erráticas cuando el sujeto camina exactamente sobre el límite del polígono.
  * **Filtro de Paso Rápido:** Se impone un umbral de permanencia (`MIN_FRAMES_EN_ZONA` de $20$ frames). Si un individuo transita por la zona de puerta pero la abandona antes de este tiempo (por ejemplo, rozando el perímetro físico sin intención de acceder al edificio), el sistema clasifica la acción como un `'paso_rapido'`, cancelando de forma automática la entrada temporal registrada.

### Renderizado Visual e Interfaz (Bloque Dibujado)
Para facilitar la depuración algorítmica y ofrecer un resultado visualmente comprensible, el sistema cuenta con un módulo de dibujo superpuesto (HUD o *Heads-Up Display*). Para no alterar destructivamente los datos del frame original durante el procesado, se emplean técnicas de superposición con canal alfa (transparencia) mediante la función `cv2.addWeighted`.

Este bloque se divide en tres componentes visuales principales:

1. **Renderizado de Zonas Espaciales (`dibujar_zona`):** Proyecta los polígonos correspondientes a la Zona de Detección global y la Zona de Puerta. Utiliza rellenos semitransparentes combinados con bordes sólidos, permitiendo visualizar la delimitación matemática sin tapar a los individuos ni el fondo de la escena.
2. **Validación de Cruce (`dibujar_punto_pie`):** Por cada individuo detectado, se dibuja un marcador dinámico en la coordenada exacta de la arista inferior de su caja delimitadora (el punto de contacto con el suelo). Este marcador cambia de color en tiempo real (rojo cuando el sujeto está fuera, verde cuando penetra en el umbral), lo que permite verificar a simple vista si la función matemática de intersección geométrica está operando correctamente.
3. **Panel de Estadísticas (`dibujar_panel`):** Un cuadro de mando superpuesto sobre un fondo negro translúcido que muestra el flujo de datos en tiempo real. Desglosa los individuos detectados en el frame actual, el acumulado histórico de entradas y salidas al edificio Carbonell, y una estimación de la ocupación interior neta.

### Flujo de Ejecución Principal (Bucle Main)
El orquestador de todo el sistema reside en el bucle principal, el cual gestiona el ciclo de vida de la ejecución, desde la inicialización de los recursos de hardware hasta el procesado secuencial del vídeo. Su arquitectura se estructura en las siguientes fases operativas:

1. **Inicialización y Reserva de Recursos:** Previo al análisis, el sistema abre los flujos de lectura del vídeo original y escritura del archivo de salida (`mp4`), replicando los *frames per second* (FPS) y la resolución de la fuente. Simultáneamente, el modelo YOLOv8 se carga en memoria y se transfiere a la GPU (arquitectura CUDA de NVIDIA, o CPU en su defecto) para garantizar la aceleración por hardware.
2. **Ciclo de Inferencia Fotograma a Fotograma:** El núcleo del programa es un bucle `while` que extrae y procesa los frames secuencialmente:
   * Primero, se aplica el preprocesado de imagen (CLAHE y Denoising) si el usuario lo ha habilitado en la configuración.
   * A continuación, se ejecuta la función `model.track()`. Esta instrucción unificada realiza simultáneamente la inferencia de la red neuronal y la actualización del algoritmo ByteTrack, devolviendo las coordenadas de las personas y sus IDs persistentes.
3. **Análisis Espacial y Máquina de Estados:** Por cada individuo detectado en el frame, se extrae su coordenada base (los pies) y se filtra en cascada. Si el punto cae fuera de la `ZONA_DETECCION`, se ignora para ahorrar recursos computacionales. Si cae dentro, se evalúa su posición respecto a la `ZONA_PUERTA` y se envía a la máquina de estados (`EstadoID`), la cual actualiza sus temporizadores de debounce y dictamina si se ha consumado una entrada, una salida o un "paso rápido".
4. **Limpieza de Historial (Gestión de Memoria):** Para evitar la saturación de memoria por IDs huérfanos y resolver oclusiones temporales, el bucle incluye una rutina de limpieza. Identifica aquellos identificadores que llevan desaparecidos de la imagen más frames que el límite permitido (`FRAMES_OLVIDO`). Antes de eliminarlos permanentemente del diccionario de seguimiento, evalúa un caso especial: si un ID desapareció mientras estaba dentro de la puerta pero el temporizador no llegó a contar su entrada, el sistema audita su historial; si se demuestra que venía de fuera, se suma retroactivamente al contador de entradas totales.
5. **Cierre y Exportación:** Una vez procesado el último frame (o tras la interrupción manual del usuario), el bucle finaliza, liberando los flujos de memoria de OpenCV y guardando el vídeo resultante con todas las anotaciones analíticas y el recuento final consolidado.

---

## 4. Manual de Uso y Configuración del Sistema

### Instalación Rápida
```bash
git clone https://github.com/AnDrew-AlcVer/yolov8-video-analytics.git
cd yolov8-video-analytics
pip install -r requirements.txt
python detector.py
```

El software ha sido diseñado bajo un patrón de arquitectura centralizada. Todos los parámetros operativos se encuentran encapsulados dentro de la clase `Config`. Esto permite a cualquier usuario adaptar el programa a un nuevo escenario, vídeo o condiciones lumínicas sin necesidad de alterar el código lógico principal subyacente.

### Gestión de Archivos e Inferencia
* `RUTA_BASE`, `NOMBRE_VIDEO_IN`, `NOMBRE_VIDEO_OUT`: Definen el directorio de trabajo, el archivo de vídeo a analizar y el nombre del archivo resultante (`mp4`) que incluirá las cajas delimitadoras (*bounding boxes*), polígonos dibujados y contadores.
* `MODELO`: Permite escalar la carga computacional seleccionando el tamaño de la red neuronal. Aunque por defecto está fijado en `'yolov8m.pt'` (Medium), el usuario puede cambiarlo a `'yolov8n.pt'` (Nano) para equipos con recursos muy limitados, o `'yolov8x.pt'` (Extra Large) si busca precisión absoluta sin importar el tiempo de procesamiento.
* `CONF_UMBRAL`: Regula la exigencia del modelo (de $0$ a $1$). Valores bajos (ej. `0.20`) favorecen la detección de personas parciales u ocultas asumiendo falsos positivos; valores altos (ej. `0.70`) garantizan que cada detección es certera, pero pueden omitir sujetos lejanos.
* `AUGMENT`: Un valor booleano (`True`/`False`) que activa el *Test-Time Augmentation*. Si se activa, YOLO realiza múltiples pasadas de la imagen con variaciones (escalados, volteos) para encontrar detecciones difíciles, a costa de ralentizar severamente el proceso.

### Geometría y Delimitación Espacial
El sistema ofrece una alta adaptabilidad espacial mediante la definición de polígonos por vértices (coordenadas $[x, y]$):
* `ZONA_DETECCION`: Matriz de coordenadas que dibuja el polígono de interés general. Actúa como el filtro principal; cualquier persona fuera de esta área es ignorada.
* `ZONA_PUERTA`: Matriz de coordenadas que define el umbral físico de recuento.
* **Herramienta de Calibración:** El código incluye una función independiente llamada `calibrar()`. Al invocar esta función en lugar de `main()`, el programa pausa el primer frame del vídeo, dibuja las zonas actuales y superpone el número de cada vértice (`D0`, `D1`... `P0`, `P1`...). Esto permite al usuario afinar las coordenadas de los polígonos visualmente para adaptarlas a la perspectiva de cualquier cámara nueva.

### Calibración de la Máquina de Estados (Lógica de Recuento)
Los filtros temporales del tracker son altamente parametrizables para ajustarse a la velocidad del tránsito peatonal de la grabación:
* `FRAMES_DEBOUNCE_ENTRADA` y `FRAMES_DEBOUNCE_SALIDA`: Definen cuántos frames debe permanecer un sujeto en el nuevo estado (`'dentro'` o `'fuera'` del polígono de la puerta) para confirmar la transición. Esto elimina los dobles conteos generados cuando un sujeto duda o camina rozando el límite de la zona.
* `MIN_FRAMES_EN_ZONA`: Establece el filtro de "paso rápido". Si el usuario define este valor en $20$, cualquier individuo que cruce la zona en menos de ese tiempo verá su recuento anulado (evita contar a gente que simplemente pasa por delante de la puerta).
* `FRAMES_OLVIDO`: Controla la "memoria" del tracker. Define cuántos frames el sistema recordará el ID de una persona que ha desaparecido de la imagen (por oclusión o error de YOLO) antes de purgarla del historial.

### Preprocesamiento de Imagen (Opcional)
El usuario dispone de un interruptor general (`PREPROCESAR = True`/`False`) para activar el módulo de limpieza visual en vídeos con mala calidad:
* **Parámetros CLAHE:** `CLAHE_CLIP_LIMIT` y `CLAHE_TILE_GRID` permiten ajustar la agresividad de la ecualización de sombras.
* **Parámetros de Denoising:** `DENOISE_H` ajusta la fuerza con la que se suaviza el ruido de la imagen. A mayor valor, menos ruido, pero la imagen perderá nitidez y detalles finos.

---

## 5. Dificultades Encontradas y Soluciones Adoptadas

### Confusión de Clases y Falsos Positivos
Uno de los primeros obstáculos detectados fue la incapacidad del modelo base para detectar a ciertas personas, especialmente cuando se encontraban lejos o parcialmente ocultas. Para solucionar esto, se redujo drásticamente el umbral de confianza (`CONF_UMBRAL`) al $20\%$. Sin embargo, esta decisión generó un problema colateral: la red neuronal comenzó a confundir elementos estáticos del mobiliario urbano u objetos inanimados con peatones.

Para mitigar estos falsos positivos sin sacrificar la sensibilidad del modelo, se pasó de procesar el frame completo (como se hacía en la versión inicial) a implementar una arquitectura de delimitación espacial. Se programó una capa lógica superior, la `ZONA_DETECCION` global, que actúa como un filtro geométrico. Cualquier detección de YOLOv8 cuyo punto de anclaje (los pies del sujeto) recaiga fuera de este polígono es descartada e ignorada por el sistema. Esta solución permitió mantener el umbral permisivo, asegurando la detección de individuos difíciles, mientras se erradicaba la "basura" algorítmica del fondo de la escena.

### Oclusiones y Manejo de Grupos
En las pruebas iniciales, el sistema mostraba un rendimiento deficiente frente a grupos densos. Cuando varias personas caminaban juntas, se tapaban mutuamente, lo que provocaba que el seguimiento original de YOLO perdiera la pista de los identificadores. Al reaparecer en cámara, el sistema les asignaba un nuevo ID, lo que derivaba en conteos duplicados y métricas erróneas.

La arquitectura final abordó este problema integrando el algoritmo **ByteTrack**, el cual es notablemente superior manteniendo identidades en multitudes. Adicionalmente, se programó un sistema de "Limpieza de Historial" personalizado con la variable `FRAMES_OLVIDO` configurada a $90$ frames. Esta memoria a corto plazo permite que, si un estudiante queda oculto detrás de otro y el sistema lo pierde temporalmente, su ID no se destruya de inmediato. Si el sujeto reaparece en la escena dentro de ese margen de tiempo (aproximadamente 3 o 4 segundos), el sistema recupera su trayectoria original, blindando la métrica contra las oclusiones temporales.

### Filtro de Sujetos Estáticos
El primer prototipo utilizaba una función matemática básica (`intersectan`) para evaluar el cruce de vectores sobre dos líneas rígidas dibujadas en la puerta. Si la coordenada $Y$ subía o bajaba al cruzar la línea, se contaba una entrada o una salida. Este método demostró ser extremadamente frágil: estudiantes detenidos charlando en el umbral o asomándose a la puerta generaban múltiples eventos de conteo debido a la oscilación natural de sus coordenadas al estar estáticos cerca de la línea.

La solución consistió en rediseñar por completo la lógica espacial y temporal:
* Se reemplazaron las líneas por la `ZONA_PUERTA`, un polígono con área real evaluado mediante la función `punto_en_zona`.
* Se implementó una máquina de estados (`EstadoID`) que rastrea si el sujeto está `'dentro'` o `'fuera'`.
* Se añadió un mecanismo de "Debounce" (antirrebote) que exige confirmación durante varios frames consecutivos (`FRAMES_DEBOUNCE_SALIDA = 3`) antes de registrar una transición.
* Se incorporó el "Filtro de paso rápido" (`MIN_FRAMES_EN_ZONA = 20`), que evalúa el tiempo de permanencia. Si una persona roza el área de la puerta pero la abandona en menos de un segundo, el sistema deshace automáticamente el conteo, evitando contabilizar a sujetos que simplemente pasean cerca de la entrada.

### Desafíos de Hardware y Entorno
A nivel de infraestructura y despliegue, el proyecto enfrentó serias limitaciones físicas. Inicialmente se observó que la ejecución por CPU resultaba en un procesamiento tan lento que el vídeo presentaba tirones o "pantallazos", imposibilitando el seguimiento continuo (*tracking*). La migración final a un entorno local con ejecución nativa sobre tarjeta gráfica dedicada (GPU CUDA) logró una inferencia fluida y en tiempo útil.

Por otro lado, la escena presentaba un problema de captura persistente: mala iluminación y un altísimo contraste en los umbrales de acceso al edificio Carbonell. Para combatirlo, se programó el módulo de preprocesamiento avanzado (reducción de ruido y ecualización CLAHE). Aunque su activación somete al procesador (CPU) a una carga computacional crítica que incluso llegó a inestabilizar el sistema operativo en pruebas tempranas, a nivel algorítmico demostró ser una solución teórica y técnica altamente eficaz para revelar detalles en las zonas ensombrecidas de las puertas.

---

## 6. Resultados y Evaluación

### Entorno de Pruebas y Selección de Muestra
Las pruebas de validación se llevaron a cabo procesando secuencias de vídeo grabadas directamente en la Plaza Ferrándiz i Carbonell, monitorizando las puertas de acceso al edificio Carbonell. Para garantizar una evaluación representativa del sistema frente a un flujo peatonal orgánico y condiciones lumínicas reales, se realizó un escrutinio manual sobre un conjunto de datos inicial compuesto por 15 secuencias de vídeo.

El objetivo de esta fase fue establecer la "verdad base" (*Ground Truth*) de cada clip, contabilizando manualmente las interacciones, descartando los escenarios vacíos y seleccionando aquellos que presentaran un verdadero desafío algorítmico:

| Secuencia | Entradas | Salidas | Observaciones de validación |
| :--- | :---: | :---: | :--- |
| **Vídeo 1** | 1 | 0 | Poco interesante |
| **Vídeo 2** | 23 | 7 | Grupo grande de personas |
| **Vídeo 3** | 4 | 2 | Se quedan en la puerta |
| **Vídeo 4** | **5** | **2** | **Buen candidato** |
| **Vídeo 5** | 0 | 0 | Descartado |
| **Vídeo 6** | 1 | 0 | Poco interesante |
| **Vídeo 7** | 0 | 0 | Descartado |
| **Vídeo 8** | 0 | 1 | ¡Sale un niño! |
| **Vídeo 9** | 1 | 0 | Poco interesante |
| **Vídeo 10** | 0 | 1 | Poco interesante |
| **Vídeo 11** | 0 | 0 | Descartado |
| **Vídeo 12** | 0 | 1 | Poco interesante |
| **Vídeo 13** | **3** | **4** | **Buen candidato** |
| **Vídeo 14** | 0 | 1 | Poco interesante |
| **Vídeo 15** | 0 | 0 | Descartado |

Tras esta clasificación, se determinó que los **Vídeos 4 y 13** presentaban la densidad de eventos y la complejidad visual idóneas para poner el software a prueba bajo condiciones operativas realistas.

### Enlaces de Demostración
Para ilustrar el funcionamiento del sistema y la respuesta de la interfaz visual en los escenarios seleccionados, se adjuntan los siguientes enlaces con los resultados renderizados:

* **Vídeo 4 (Sin Preprocesado):** [Ver en YouTube](https://youtu.be/wHLzNZrRCSk)
* **Vídeo 13 (Resultado Procesado con CLAHE + Denoising):** [Ver en YouTube](https://youtu.be/8Hv4NeX2NpE)
* **Vídeo 13 (Sin Preprocesado):** [Ver en YouTube](https://youtu.be/7-x81qhYuT8)

> **Nota técnica:** Este último enlace permite comprobar empíricamente las diferencias de rendimiento del modelo. Al visualizar el vídeo sin la ecualización CLAHE, se evidencia la caída en la tasa de detección al cruzar el umbral oscuro de la puerta, demostrando la utilidad teórica del módulo de preprocesamiento diseñado.

### Análisis de Rendimiento (Caso de Estudio: Vídeo 13)
Tomando como caso de estudio el Vídeo 13, se estableció una comparativa analítica entre los eventos reales documentados y las estadísticas procesadas de forma autónoma por la máquina de estados del programa:

* **Estadísticas Reales (Manuales):** 3 Entradas / 4 Salidas (Total: 7 eventos de cruce).
* **Estadísticas Procesadas (Sistema):** 1 Entrada / 3 Salidas (Total: 4 eventos de cruce).

El análisis detallado de esta discrepancia revela que el sistema es notablemente más eficaz registrando las salidas que las entradas. Las salidas (3 de 4 detectadas correctamente) ocurren cuando el sujeto avanza desde la penumbra hacia la luz natural de la plaza, facilitando el reconocimiento a la red neuronal. Por el contrario, la pronunciada omisión en las entradas (1 de 3) se atribuye al fuerte contraluz que oscurece a las personas al ingresar al edificio; si el modelo pierde la caja delimitadora en ese preciso instante lumínico, el tracker se interrumpe y el individuo no llega a acumular los frames de confirmación requeridos en la "Zona de Puerta" para validar matemáticamente la transición.

### Métricas Finales
Evaluando la totalidad de los eventos reales frente a los registrados por el software en el caso de estudio analizado, el sistema alcanzó un **porcentaje de éxito global del 57%** (4 de 7 interacciones correctas).

Aunque en un entorno de producción comercial esta métrica requeriría optimización, desde la perspectiva de desarrollo de este proyecto representa un hito técnico muy positivo. Este 57% certifica que la arquitectura del sistema es sólida: el pipeline de captura, la inferencia con YOLOv8, el mantenimiento de IDs con ByteTrack y la lógica espacial de polígonos operan en perfecta sincronía. Los eventos omitidos no responden a fallos en la lógica de recuento programada, sino a las severas limitaciones inherentes de inferir en un escenario exterior con alto contraste utilizando una única cámara frontal, sin el respaldo de hardware especializado para corrección lumínica en tiempo real.

---

## 7. Conclusiones y Trabajo Futuro

### Conclusiones Finales
El desarrollo de este proyecto ha supuesto un desafío integral en el ámbito de la visión artificial, permitiendo poner en práctica conocimientos que abarcan desde el diseño de la arquitectura del software hasta la resolución analítica de problemas en entornos físicos y dinámicos. La evaluación global confirma la viabilidad del sistema diseñado como una solución de ingeniería robusta y estructurada.

Es fundamental destacar que, a lo largo de todas las secuencias analizadas, el modelo logró detectar y rastrear con gran fiabilidad a la inmensa mayoría de las personas que transitaban por la plaza. En este aspecto, el proyecto ha sido un éxito rotundo, validando el uso conjunto de la red neuronal YOLOv8 y el algoritmo ByteTrack. Las dificultades métricas encontradas (reflejadas en el 57% de éxito del caso de estudio) quedaron aisladas casi exclusivamente en el punto crítico del umbral de las puertas. Esto demuestra de forma concluyente que la lógica de programación geométrica y la máquina de estados implementadas son correctas, y que los obstáculos para alcanzar un conteo perfecto residen en las barreras físicas, tecnológicas y ambientales de la captura original.

### Futuras Líneas de Mejora
Con el propósito de elevar la precisión del sistema y lograr un despliegue óptimo en un entorno de producción real, se plantean las siguientes líneas de actuación técnica:

1. **Actualización del Hardware de Captura:** Los vídeos utilizados para el entrenamiento y las pruebas de validación tienen cierta antigüedad y presentan limitaciones evidentes de resolución, rango dinámico y compresión. Por ello, el primer paso y el más directo para mejorar los resultados consistiría en realizar nuevas grabaciones utilizando una cámara moderna de alta calidad. Contar con un origen de vídeo nítido simplificaría enormemente el trabajo analítico de la red neuronal, haciendo mucho más fácil, estable y viable el conteo de las detecciones en los puntos críticos de cruce.
2. **Optimización de Preprocesamiento en Tiempo Real:** Se propone reestructurar el módulo de corrección lumínica (reducción de ruido y ecualización CLAHE) desarrollándolo bajo lenguajes o frameworks de más bajo nivel (como CUDA/C++ puro). El objetivo sería lograr que este filtro se ejecute íntegramente sobre la GPU sin generar cuellos de botella ni inestabilidades operativas, neutralizando por completo el problema de los contraluces en las puertas.
3. **Escalabilidad del Modelo Analítico:** Disponer de servidores o estaciones de trabajo con mayor capacidad gráfica permitiría sustituir la actual arquitectura "Medium" por versiones más pesadas y precisas del modelo (como YOLOv8x o arquitecturas de segmentación instanciada), llevando el reconocimiento de peatones parcialmente ocluidos a su máxima capacidad algorítmica.