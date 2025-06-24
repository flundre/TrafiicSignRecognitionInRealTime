# Traffic Sign Recognition in Real Time

Цей проєкт призначений для розпізнавання дорожніх знаків у режимі реального часу за допомогою згорткової нейронної мережі (CNN), створеної на основі бібліотеки TensorFlow. Модель отримує зображення з вебкамери, визначає знак на відеопотоці та виводить передбачення на екран.

## Що потрібно для запуску

Перед запуском необхідно встановити Python (рекомендовано версія 3.8 або новіше) та встановити необхідні залежності.

### Крок 1. Клонування репозиторію

```bash
git clone https://github.com/flundre/TrafiicSignRecognitionInRealTime.git
cd TrafiicSignRecognitionInRealTime
```

### Крок 2. Створення віртуального середовища (необов’язково, але рекомендовано)

#### Windows:
```bash
python -m venv venv
venv\Scripts\activate
```

#### Linux / macOS:
```bash
python3 -m venv venv
source venv/bin/activate
```

### Крок 3. Встановлення залежностей

У проєкті є файл `requirements.txt`, виконайте:

```bash
pip install -r requirements.txt
```

## Робота програми

Для створення\редагування\тренування моделі див. generating_model.ipynb

Для режиму роботи з одним зображенням

```bash
python predict_image.py
```

Для режиму роботи в реальному часі (необхідна вебкамера)

```bash
python cv.py
```

