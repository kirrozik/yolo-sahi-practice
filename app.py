import os
import tempfile
import gradio as gr
from sahi import AutoDetectionModel
from sahi.predict import get_sliced_prediction
from PIL import Image
import requests
import time

AVAILABLE_MODELS = {
    "YOLOv8 Nano (быстрая)": "yolov8n.pt",
    "YOLOv8 Small (баланс)": "yolov8s.pt",
    "YOLOv8 Medium (точная)": "yolov8m.pt",
    "YOLOv8 Large (очень точная)": "yolov8l.pt",
    "YOLOv8 X-Large (максимальная)": "yolov8x.pt",
    "YOLO11 Nano (новая, быстрая)": "yolo11n.pt",
    "YOLO11 Small (новая, баланс)": "yolo11s.pt",
    "YOLO11 Medium (новая, точная)": "yolo11m.pt",
    "YOLO11 Large (новая, очень точная)": "yolo11l.pt",
    "YOLO11 X-Large (новая, максимальная)": "yolo11x.pt",
}

MODEL_URLS = {
    "yolov8n.pt": "https://github.com/ultralytics/assets/releases/download/v8.2.0/yolov8n.pt",
    "yolov8s.pt": "https://github.com/ultralytics/assets/releases/download/v8.2.0/yolov8s.pt",
    "yolov8m.pt": "https://github.com/ultralytics/assets/releases/download/v8.2.0/yolov8m.pt",
    "yolov8l.pt": "https://github.com/ultralytics/assets/releases/download/v8.2.0/yolov8l.pt",
    "yolov8x.pt": "https://github.com/ultralytics/assets/releases/download/v8.2.0/yolov8x.pt",
    "yolo11n.pt": "https://github.com/ultralytics/assets/releases/download/v8.3.0/yolo11n.pt",
    "yolo11s.pt": "https://github.com/ultralytics/assets/releases/download/v8.3.0/yolo11s.pt",
    "yolo11m.pt": "https://github.com/ultralytics/assets/releases/download/v8.3.0/yolo11m.pt",
    "yolo11l.pt": "https://github.com/ultralytics/assets/releases/download/v8.3.0/yolo11l.pt",
    "yolo11x.pt": "https://github.com/ultralytics/assets/releases/download/v8.3.0/yolo11x.pt",
}

MODEL_DESCRIPTIONS = {
    "yolov8n.pt": "Самая быстрая, 6MB, mAP 37.3. Идеальна для тестов.",
    "yolov8s.pt": "Баланс скорости и точности, 22MB, mAP 44.9.",
    "yolov8m.pt": "Средняя точность, 52MB, mAP 50.2.",
    "yolov8l.pt": "Высокая точность, 88MB, mAP 52.9.",
    "yolov8x.pt": "Максимальная точность, 137MB, mAP 53.9.",
    "yolo11n.pt": "Новая версия 2024, улучшенная архитектура, 6MB.",
    "yolo11s.pt": "YOLO11 Small, улучшенные результаты, 22MB.",
    "yolo11m.pt": "YOLO11 Medium, отличная точность, 52MB.",
    "yolo11l.pt": "YOLO11 Large, высокая точность, 88MB.",
    "yolo11x.pt": "YOLO11 X-Large, максимальная точность, 137MB.",
}

current_model = None
current_model_path = None
current_device = None

def download_model(model_path, url, max_retries=3, timeout=10):
    if os.path.exists(model_path):
        return True, f"Модель уже существует: {model_path}"

    print(f"Начинаем скачивание модели: {url}")

    for attempt in range(max_retries):
        try:
            print(f"Попытка {attempt + 1}/{max_retries}...")
            temp_path = model_path + ".tmp"

            response = requests.get(url, stream=True, timeout=timeout)
            response.raise_for_status()

            total_size = int(response.headers.get('content-length', 0))
            downloaded_size = 0

            with open(temp_path, 'wb') as f:
                for chunk in response.iter_content(chunk_size=8192):
                    if chunk:
                        f.write(chunk)
                        downloaded_size += len(chunk)

                        if total_size > 0:
                            percent = (downloaded_size / total_size) * 100
                            print(f"\rСкачано: {percent:.1f}% ({downloaded_size}/{total_size} bytes)", end="")

            print()
            os.rename(temp_path, model_path)

            return True, f"Модель успешно скачана: {model_path}"

        except requests.exceptions.Timeout:
            print(f"Таймаут при попытке {attempt + 1}")
            if attempt < max_retries - 1:
                time.sleep(2)
        except requests.exceptions.RequestException as e:
            print(f"Ошибка сети: {e}")
            if attempt < max_retries - 1:
                time.sleep(2)
        except Exception as e:
            print(f"Неожиданная ошибка: {e}")
            if os.path.exists(temp_path):
                os.remove(temp_path)
            return False, f"Ошибка скачивания: {str(e)}"

    return False, f"Не удалось скачать модель после {max_retries} попыток"

def load_model(model_path, confidence_threshold=0.3, device="cpu"):
    global current_model, current_model_path, current_device

    if current_model_path == model_path and current_device == device:
        return f"Модель уже загружена: {os.path.basename(model_path)} на {device}"

    if not os.path.exists(model_path):
        model_name = os.path.basename(model_path)
        url = MODEL_URLS.get(model_name)

        if not url:
            return f"Ошибка: Нет URL для скачивания {model_name}"

        success, download_status = download_model(model_path, url)

        if not success:
            return f"Не удалось скачать модель: {download_status}"

    print(f"Инициализация модели: {model_path} на устройстве {device}...")
    try:
        current_model = AutoDetectionModel.from_pretrained(
            model_type="ultralytics",
            model_path=model_path,
            confidence_threshold=confidence_threshold,
            device=device,
        )
        current_model_path = model_path
        current_device = device
        model_name = os.path.basename(model_path)
        description = MODEL_DESCRIPTIONS.get(model_path, "Пользовательская модель")
        return f"Успешно загружена: {model_name} на {device}\n{description}"
    except Exception as e:
        return f"Ошибка инициализации модели: {str(e)}"

def predict(image,
            model_choice,
            custom_model_path,
            device_choice,
            slice_height=512,
            slice_width=512,
            overlap_height_ratio=0.2,
            overlap_width_ratio=0.2,
            confidence_threshold=0.3):
    global current_model

    if image is None:
        return None, "Нет изображения", ""

    if custom_model_path and os.path.exists(custom_model_path):
        model_path = custom_model_path
    else:
        model_path = AVAILABLE_MODELS.get(model_choice, "yolov8n.pt")

    device = "cuda:0" if device_choice == "CUDA (GPU)" else "cpu"
    status = load_model(model_path, confidence_threshold, device)

    if current_model is None or "Ошибка" in status or "Не удалось" in status:
        return None, "Ошибка выполнения", status

    with tempfile.TemporaryDirectory() as temp_dir:
        input_path = os.path.join(temp_dir, "input.jpg")
        image.save(input_path)

        print("Запуск SAHI инференса...")
        result = get_sliced_prediction(
            input_path,
            current_model,
            slice_height=slice_height,
            slice_width=slice_width,
            overlap_height_ratio=overlap_height_ratio,
            overlap_width_ratio=overlap_width_ratio,
        )

        num_objects = len(result.object_prediction_list)

        result.export_visuals(export_dir=temp_dir, file_name="output")

        output_path = os.path.join(temp_dir, "output.png")
        if not os.path.exists(output_path):
            output_path = os.path.join(temp_dir, "output.jpg")

        model_name = os.path.basename(model_path)
        stats = f"Модель: {model_name}\nУстройство: {device}\nНайдено объектов: {num_objects}"

        return Image.open(output_path), stats, status

with gr.Blocks(title="YOLO + SAHI Детектор", theme=gr.themes.Soft()) as demo:
    gr.Markdown("# YOLO + SAHI Детектор объектов")
    gr.Markdown("Загрузите изображение разрешенем до 4K для детекции мелких объектов")
    gr.Markdown("**Поддерживаемые модели:** YOLOv8 и YOLO11 (новейшая версия 2024)")

    with gr.Row():
        with gr.Column(scale=1):
            input_image = gr.Image(type="pil", label="Загрузите изображение")

            with gr.Accordion("Выбор модели YOLO", open=True):
                model_choice = gr.Dropdown(
                    choices=list(AVAILABLE_MODELS.keys()),
                    value="YOLOv8 Small (баланс)",
                    label="Предустановленные модели"
                )
                custom_model_path = gr.Textbox(
                    label="Или путь к своей модели (.pt файл)",
                    placeholder="/home/user/best.pt",
                    info="Если указано, будет использована эта модель вместо выбранной выше"
                )

            with gr.Accordion("Выбор устройства", open=True):
                device_choice = gr.Radio(
                    choices=["CPU (процессор)", "CUDA (GPU)"],
                    value="CPU (процессор)",
                    label="Устройство для обработки",
                    info="CUDA работает только если есть видеокарта NVIDIA и установлены драйверы"
                )

            with gr.Accordion("Настройки SAHI", open=False):
                slice_height = gr.Slider(256, 1024, value=512, step=256,
                                        label="Высота слайса (pixels)")
                slice_width = gr.Slider(256, 1024, value=512, step=256,
                                       label="Ширина слайса (pixels)")
                overlap_h = gr.Slider(0.0, 0.5, value=0.2, step=0.1,
                                     label="Перекрытие по высоте (ratio)")
                overlap_w = gr.Slider(0.0, 0.5, value=0.2, step=0.1,
                                     label="Перекрытие по ширине (ratio)")

            with gr.Accordion("Настройки детекции", open=False):
                conf_threshold = gr.Slider(0.1, 0.9, value=0.3, step=0.05,
                                          label="Порог уверенности (confidence)")

            submit_btn = gr.Button("Запустить детекцию", variant="primary", size="lg")

        with gr.Column(scale=1):
            output_image = gr.Image(type="pil", label="Результат детекции")
            stats_text = gr.Textbox(label="Статистика", lines=3)
            model_status = gr.Textbox(label="Статус модели", lines=3)

    gr.Examples(
        examples=["test_image.jpg"] if os.path.exists("test_image.jpg") else [],
        inputs=input_image
    )

    submit_btn.click(
        fn=predict,
        inputs=[
            input_image,
            model_choice,
            custom_model_path,
            device_choice,
            slice_height,
            slice_width,
            overlap_h,
            overlap_w,
            conf_threshold
        ],
        outputs=[output_image, stats_text, model_status]
    )

if __name__ == "__main__":
    demo.launch(server_name="0.0.0.0", server_port=7860)
