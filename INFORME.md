# Predicción de demanda eléctrica con LSTM

Se predice la demanda de la hora siguiente, en MW, a partir de una ventana de horas ya observadas. La serie va del 1 de enero de 2025 al 31 de diciembre de 2026. TensorFlow no tiene rueda para Python 3.14 en esta máquina; las redes están en PyTorch (`nn.LSTM`), con el mismo diseño de celdas LSTM.

## Limpieza

El archivo trae 17550 filas y 17520 horas distintas: el calendario horario está completo, pero desordenado. Se limpia antes de formar cualquier secuencia. Una medición anómala o faltante se elimina. No se imputa, no se interpola y no se recorta al límite.

| Regla | Filas |
| --- | ---: |
| Duplicados exactos | 30 |
| Marcas repetidas con valores distintos | 0 |
| Faltantes en temperatura, humedad, viento o precio | 279 |
| Demanda fuera de 300–1050 MW | 23 |
| Temperatura fuera de 0–40 °C | 24 |
| Humedad fuera de 0–100 % | 9 |
| Viento negativo o mayor de 80 km/h | 22 |
| Precio no positivo o mayor de 60 | 9 |
| Unión de faltantes y anomalías | 365 |
| Horas que quedan | 17155 |
| Huecos de más de una hora | 355 |

La demanda del grueso de la serie vive entre unos 350 y 970 MW. Por debajo de 300 MW y por encima de 1050 MW hay picos aislados que a la hora siguiente vuelven a 400–800 MW: no son una rampa real. El clima observado se concentra entre unos 4 °C y 33 °C; bajo 0 °C o sobre 40 °C no hay continuidad con esa masa. Humedad fuera de 0–100 % y viento negativo son imposibles. El viento por encima de 80 km/h queda separado del tramo continuo (hasta unos 67 km/h). El precio por encima de 60 queda fuera de la banda observada (el percentil 99.9 está cerca de 42).

`hora`, `dia_semana`, `mes` y `fin_semana` coinciden con la marca de tiempo. `periodo_dia` trae mayúsculas, espacios y la palabra «mañana» con distinta codificación; 60 etiquetas no coinciden con la hora. El periodo se recalcula desde la hora, porque es un atributo del reloj y no una medición de sensor.

La columna `demanda_objetivo` coincide con la demanda de la hora siguiente en 16782 de 16799 pares consecutivos ya limpios. Los 17 que difieren se descartan como etiqueta: el objetivo del modelo es la `demanda_mw` observada en t+1. Esa columna no entra como variable.

## Diseño del experimento

La serie se ordena y después se corta en el tiempo, 70 % / 15 % / 15 %:

| Tramo | Inicio | Fin | Horas |
| --- | --- | --- | ---: |
| Entrenamiento | 2025-01-01 00:00 | 2026-05-26 20:00 | 12008 |
| Validación | 2026-05-26 21:00 | 2026-09-13 09:00 | 2573 |
| Prueba | 2026-09-13 10:00 | 2026-12-31 23:00 | 2574 |

No se usa un `train_test_split` aleatorio. Una secuencia pertenece a un tramo si la hora que se predice cae en ese tramo. Las horas de entrada pueden venir del tramo anterior: son pasado. Ninguna secuencia de entrenamiento tiene su objetivo en validación o en prueba. Si entre las horas de la ventana hay un hueco, la secuencia no se crea.

La ventana `[t-n, t-1]` predice la demanda en `t`. No entran el clima, el precio ni la demanda de `t`.

Cada hora se representa con la demanda, el clima, el precio, fin de semana, festivo, seno y coseno de hora, día y mes, y el periodo del día en cuatro indicadores. El seno y el coseno evitan que la hora 23 y la hora 0 queden lejos en la escala numérica.

`StandardScaler` se ajusta solo con las horas de entrenamiento. La demanda objetivo se estandariza con la media y la desviación de ese mismo tramo. Las métricas se calculan después de volver a MW.

El entrenamiento usa Adam con tasa 0.001, pérdida MSE sobre la demanda estandarizada, lotes de 256, recorte de gradiente 1.0, máximo 30 épocas y early stopping con paciencia 6. Las curvas de pérdida se miden con dropout apagado, para que entrenamiento y validación sean comparables. La columna Épocas de la tabla es la época restaurada.

## Arquitecturas

**A. LSTM base.** Una LSTM de 64 unidades, dropout 0.2 y una capa lineal. 21825 parámetros. El último estado resume la ventana.

**B. LSTM profunda.** Tres LSTM apiladas (128, 64 y 32), con dropout 0.3, 0.3 y 0.2, una capa densa de 16 unidades con ReLU y la salida. 139041 parámetros. Cada capa lee la secuencia que produjo la anterior.

**C. LSTM con atención.** Una LSTM de 96 unidades, dropout 0.3, atención aditiva sobre todas las horas de la ventana, concatenación de ese contexto con el último estado, capa densa de 32 y dropout 0.2. 60545 parámetros. La idea es que una hora lejana —por ejemplo la misma hora del día anterior, en una ventana de 24 o 48 h— pueda seguir pesando, en lugar de depender solo del estado final.

## Resultados en prueba

| Modelo | Ventana | MAE | MSE | RMSE | MAPE | R² | Épocas |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| LSTM base | 12 | 19.49 | 608.98 | 24.68 | 3.33 | 0.9078 | 27 |
| LSTM profunda | 12 | 19.43 | 605.84 | 24.61 | 3.32 | 0.9083 | 19 |
| LSTM con atención | 12 | 19.66 | 612.80 | 24.75 | 3.36 | 0.9072 | 19 |
| LSTM base | 24 | 20.58 | 673.58 | 25.95 | 3.51 | 0.8968 | 29 |
| LSTM profunda | 24 | 20.91 | 709.05 | 26.63 | 3.60 | 0.8913 | 14 |
| LSTM con atención | 24 | 20.28 | 659.73 | 25.69 | 3.48 | 0.8989 | 26 |
| LSTM base | 48 | 21.14 | 717.53 | 26.79 | 3.64 | 0.8880 | 21 |
| LSTM profunda | 48 | 21.02 | 722.33 | 26.88 | 3.62 | 0.8873 | 18 |
| LSTM con atención | 48 | 20.28 | 671.24 | 25.91 | 3.51 | 0.8952 | 25 |

Persistencia (la hora siguiente igual a la última), en las mismas horas de prueba de la ventana de 12 h: MAE 35.56 MW, RMSE 44.71 MW, MAPE 6.01 %, R² 0.697.

Las gráficas están en `artifacts/charts/`: pérdida de entrenamiento y validación, demanda real frente a predicha, error de predicción, comparación de modelos y pesos de atención.

## Preguntas

### 1. ¿Por qué LSTM es apropiada para este problema?

La demanda de la hora siguiente depende de la evolución reciente: ciclo diario, ciclo semanal e inercia de pocas horas. La LSTM recorre la ventana y conserva, mediante sus compuertas, la parte de ese pasado que sigue siendo útil. En cada hora entran clima, precio, calendario y demanda, sin forzar un efecto lineal. Frente a repetir la última demanda, el MAE de prueba baja de 35.56 MW a 19.49 MW.

### 2. ¿Qué efecto tiene cambiar la ventana temporal?

Pasar de 12 a 24 y a 48 horas sube el MAE de la LSTM base de 19.49 a 20.58 y 21.14 MW. Doce horas ya contienen el ciclo que determina la hora siguiente. La ventana larga reduce las secuencias útiles (de 9294 a 4084 en entrenamiento) porque un hueco rompe la racha, y obliga al estado final a recordar horas que aportan poco. La atención mejora la ventana de 48 h (MAE 20.28 MW) y sigue por detrás de los modelos de 12 h.

### 3. ¿Existe sobreajuste?

El MAE de entrenamiento y el de prueba se mantienen juntos. En la base de 12 h son 19.54 MW y 19.49 MW; en la profunda, 19.65 MW y 19.43 MW. El hueco más grande es de unas 0.4 MW en la base de 48 h. Las curvas bajan en paralelo y early stopping corta donde la validación deja de mejorar. El R² de entrenamiento (cerca de 0.93) supera un poco al de prueba (0.91) porque el tramo de entrenamiento abarca más estaciones; en MW el error se sostiene fuera del entrenamiento.

### 4. ¿Qué es y qué efecto tiene Dropout?

Dropout apaga una fracción aleatoria de neuronas en cada paso de entrenamiento, para que la red reparta lo que aprende. En evaluación participan todas. Aquí la tasa es 0.2 en la base y llega a 0.3 en las capas profundas y en la secuencia de la atención. Con eso, la pérdida de entrenamiento —medida sin dropout— no se separa de la validación, incluso en la red de 139 mil parámetros.

### 5. ¿Qué es y qué efecto tiene Early Stopping?

Vigila el MSE de validación. Si no mejora durante 6 épocas, se detiene el entrenamiento y se restauran los pesos de la mejor época, con un tope de 30. Seis modelos pararon antes del tope: la profunda de 12 h se quedó en la época 19 y la de 24 h en la 14. La tabla reporta esa época restaurada. El pronóstico publicado es el de mejor validación, no el de la última pasada.

### 6. ¿Más neuronas implican necesariamente mejores resultados?

No. La profunda (139041 parámetros) gana 0.05 MW de MAE en 12 h y es la peor arquitectura en 24 h. En 48 h la atención, con menos parámetros, la supera. Una capa de 64 unidades ya captura el paso de una hora.

### 7. ¿Qué modelo recomendaría para producción y por qué?

La LSTM base con ventana de 12 horas: MAE 19.49 MW, RMSE 24.68 MW, MAPE 3.33 % y R² 0.9078. La profunda de 12 h tiene el menor RMSE del tablero (24.61 MW), a 0.06 MW de la base. Ese margen no cambia la operación, y la base tiene seis veces menos parámetros y una ventana que se puede completar con más frecuencia en una serie con huecos. Corta casi a la mitad el error de la persistencia.

### 8. ¿Qué variables pueden influir más en la demanda?

Al permutar bloques en la prueba del modelo recomendado, el calendario sube el MAE 55.3 MW, el clima 12.9 MW, la demanda reciente 1.0 MW y el precio 0.5 MW. Dentro del calendario pesan el periodo del día (+30.0 MW) y el fin de semana (+20.0 MW). La hora sola pesa menos (+3.6 MW) porque el periodo ya describe el reloj. Radiación (+4.0 MW) y temperatura (+3.3 MW) son las meteorológicas relevantes. Es lo que usa el modelo, no una causa: el calendario fija el perfil de actividad y el clima lo corrige.

### 9. ¿Qué limitaciones tiene el modelo?

Predice una sola hora y no ve el clima ni el precio de esa hora. En prueba la mediana del error absoluto es 16.1 MW, el percentil 95 es 48 MW y el máximo 93 MW, con más error en las horas 5, 7 y 12. Hacen falta 12 horas seguidas; los huecos de la limpieza rompen ventanas porque no se inventaron valores. La prueba es solo del 13 de septiembre al 31 de diciembre de 2026. La atención de 48 h reparte el peso casi uniforme, cerca de 1/48, y no se concentra en un rezago concreto.

## App

Desde esta carpeta:

```bash
python train.py
python app.py
```

La aplicación queda en http://127.0.0.1:5050. Ahí se elige el modelo, se editan las horas de la ventana y se lee la demanda predicha de la hora siguiente. La ventana histórica que aparece al abrir es el último bloque consecutivo de la serie limpia.
