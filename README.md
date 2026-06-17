# YOLO + SAHI Inference

Проект для летней практики. Используется библиотека Ultralytics для инференса YOLOv8 и SAHI для детекции мелких объектов на изображениях высокого разрешения через веб-интерфейс.

## Установка (WSL / Linux)

```bash
python3 -m venv venv
source venv/bin/activate
pip install -r requirements.txt
```

Посе того как все установиться выполняем 
```bash
python app.py
```
и будет написан адрес доступа, если не открывается тот что написано можно попробовать http://127.0.0.1:7860/
