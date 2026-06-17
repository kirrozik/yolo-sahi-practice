import os
import tempfile
import gradio as gr
from sahi import AutoDetectionModel
from sahi.predict import get_sliced_prediction
from PIL import Image

print("Загрузка модели YOLO + SAHI...")
# ИЗМЕНЕНИЕ 1: from_pretrained вместо from_model_type
detection_model = AutoDetectionModel.from_pretrained(
    model_type="ultralytics",
    model_path="yolov8n.pt",  # Замени на best.pt когда обучишь свою модель
    confidence_threshold=0.6,
    device="cpu"  # Поставь "cuda:0" если есть видеокарта NVIDIA
)
print("Модель загружена!")

def predict(image):
    """
    Принимает PIL Image из Gradio, прогоняет через SAHI
    и возвращает PIL Image с нарисованными боксами.
    """
    if image is None:
        return None
    
    # Создаем временную директорию, которая сама удалится после выполнения
    with tempfile.TemporaryDirectory() as temp_dir:
        # SAHI надежнее работает с путями к файлам, поэтому сохраняем входную картинку
        input_path = os.path.join(temp_dir, "input.jpg")
        image.save(input_path)
        
        # Запускаем SAHI
        result = get_sliced_prediction(
            input_path,
            detection_model,
            slice_height=512,
            slice_width=512,
            overlap_height_ratio=0.2,
            overlap_width_ratio=0.2,
        )
        
        # ИЗМЕНЕНИЕ 2: Сохраняем визуализацию с боксами во временную папку
        result.export_visuals(export_dir=temp_dir, file_name="output")
        
        # Открываем сохраненную картинку, чтобы отдать её Gradio
        output_path = os.path.join(temp_dir, "output.png")
        # На случай, если SAHI сохранил в jpg (зависит от версии)
        if not os.path.exists(output_path):
            output_path = os.path.join(temp_dir, "output.jpg")
            
        return Image.open(output_path)

# Создаем веб-интерфейс
demo = gr.Interface(
    fn=predict,
    inputs=gr.Image(type="pil", label="Загрузите изображение"),
    outputs=gr.Image(type="pil", label="Результат детекции (SAHI)"),
    title="YOLO + SAHI Детектор",
    description="Загрузите фото для детекции объектов. Используется SAHI для поиска мелких объектов.",
    examples=["test_image.jpg"] if os.path.exists("test_image.jpg") else []
)

if __name__ == "__main__":
    demo.launch(server_name="0.0.0.0", server_port=7860)
