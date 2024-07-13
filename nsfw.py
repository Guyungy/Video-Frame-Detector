import cv2
import os
from transformers import AutoImageProcessor, AutoModelForImageClassification
from PIL import Image
import torch

# 加载处理器和模型
processor = AutoImageProcessor.from_pretrained("Falconsai/nsfw_image_detection")
model = AutoModelForImageClassification.from_pretrained("Falconsai/nsfw_image_detection")

# 定义类别名称（假设类别标签已知）
labels = ["safe_for_work", "not_safe_for_work"]

# 指定目录路径
root_dir = r"H:\DouyinLive\悲伤西红柿🍅"

# 支持的视频文件扩展名
video_extensions = ['.mp4', '.mov', '.avi', '.mkv']

# 遍历指定目录及其子目录中的所有视频文件
for subdir, dirs, files in os.walk(root_dir):
    for file in files:
        if any(file.lower().endswith(ext) for ext in video_extensions):
            video_path = os.path.join(subdir, file)
            output_dir = subdir
            os.makedirs(output_dir, exist_ok=True)

            # 打开视频文件
            cap = cv2.VideoCapture(video_path)
            frame_count = 0

            while cap.isOpened():
                ret, frame = cap.read()
                if not ret:
                    break

                frame_count += 1
                # 每间隔一定帧数进行一次检测
                if frame_count % 30 == 0:  # 可以根据需要调整间隔帧数
                    # 将帧转换为PIL图像
                    image = Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))

                    # 处理图像
                    inputs = processor(images=image, return_tensors="pt")

                    # 进行预测
                    with torch.no_grad():
                        outputs = model(**inputs)

                    # 获取预测结果
                    logits = outputs.logits
                    predicted_class_idx = logits.argmax(-1).item()

                    # 打印预测结果
                    print(f"Video {video_path}, Frame {frame_count} - Predicted class: {labels[predicted_class_idx]}")

                    # 如果是NSFW内容，保存截图
                    if labels[predicted_class_idx] == "not_safe_for_work":
                        screenshot_path = os.path.join(output_dir, f"{os.path.splitext(file)[0]}_nsfw_frame_{frame_count}.jpg")
                        try:
                            image.save(screenshot_path)
                            print(f"NSFW frame saved at: {screenshot_path}")
                        except Exception as e:
                            print(f"Error saving frame {frame_count}: {e}")

            cap.release()

cv2.destroyAllWindows()
