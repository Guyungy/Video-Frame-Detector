下面是这个脚本的详细中文README文档，说明如何使用该脚本。

# NSFW 视频帧检测器

该脚本遍历指定目录及其子目录中的所有视频文件，检测视频中的NSFW（不适合工作场所）内容，并保存这些NSFW帧的截图。它使用Transformers库中的预训练模型进行NSFW图像分类。

## 环境要求

- Python 3.7+
- Transformers
- Pillow
- Torch
- OpenCV

## 安装

使用以下命令安装所需的库：

```bash
pip install transformers pillow torch opencv-python
```

## 使用方法

1. **克隆或下载脚本**到你的本地机器。

2. **更新脚本中的目录路径**，将`root_dir`变量设置为你想要遍历的根目录。例如：

   ```python
   root_dir = r"c:\Users\diriw\Documents\code\临时"
   ```

3. **运行脚本**，它将遍历指定目录及其子目录中的所有视频文件，并进行NSFW内容检测。如果检测到NSFW内容，将保存相应的截图到与视频文件相同的目录中。

## 脚本说明

脚本 `nsfw_video_detector.py` 的代码如下：

```python
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
root_dir = r"c:\Users\diriw\Documents\code\临时"

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
```

## 注意事项

- 确保在运行脚本前，指定的目录路径是正确的，并且程序具有在该目录中创建文件的权限。
- 根据需要调整检测间隔（例如，每30帧检测一次）以平衡检测频率和性能。
- 确保安装了所有必要的库和依赖项。

通过这些步骤，你可以使用该脚本在指定目录及其子目录中检测视频文件的NSFW内容，并保存相关的截图。