# Puppeteers

# Кейс "Выявление нетипичных операций по транзакционной активности"

## 1. [clear.ipynb](clear.ipynb) - Очистка данных

## 2. [model.ipynb](model.ipynb) - Содержит код создания модели ML

### 1.2 [use_model.py](use_model.py) - При вызове use_model(путь_к_тест_файлу) будет использована ML модель для поиска нетипичных транзакций после чего результат будет сохранен в файл "preds_use_model.csv"

## 3. [EDA.ipynb](EDA.ipynb) - Исследовательский анализ данных

## 4. [anomaly.ipynb](anomaly.ipynb) - Паттерны поведения (пункт 3 и 4 из ТЗ)

### 4.1 [map.html](map.html) - Карта местоположения откуда происходила подозрительная активность

# Результат метрики модели при проверки через предоставленную программу
![img_1.png](img/img_1.png)

# Алгоритм создания модели

### 1. Предобработка категориальных данных:<br>

Для категориальных признаков выполняется преобразование редких значений в категорию "SMALL" (если встречаются редко,
меньше заданного порога).
Затем каждый категориальный столбец кодируется с помощью Label Encoding.
<br>
<br>

### 2. Обнаружение аномалий:<br>

Каждый из методов оценивает аномальности, присваивая значения, которые затем комбинируются для финального списка
аномальных данных.
Для поиска аномалий используются три алгоритма:

1. Isolation Forest
2. Local Outlier Factor (LOF)
3. Elliptic Envelope.
   <br>
   <br>

### 3. Создание метки для аномалий:<br>

На основе решений всех алгоритмов создается бинарная метка anomaly для каждой строки данных.
<br>
<br>

### 4. Балансировка классов:<br>

Для обучения модели на основе меток аномалий используется метод SMOTE, который генерирует дополнительные примеры для
редкого класса (аномальных данных), чтобы сбалансировать классы.
<br>
<br>

### 5. Обучение модели классификации:<br>

Для классификации аномалий используется CatBoostClassifier, который обучается на сбалансированном наборе данных.
<br>
<br>

### 6. Оценка модели:<br>

После обучения модель оценивается с помощью метрики classification_report<br>
**Модель демонстрирует высокое качество с точностью 98%, сбалансированными показателями для обоих классов и отличной способностью выявлять как нормальные, так и аномальные данные.**
![img.png](img/img.png)