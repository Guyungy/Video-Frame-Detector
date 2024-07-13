import cv2
import os
import threading
from queue import Queue, Empty
from transformers import AutoImageProcessor, AutoModelForImageClassification
from PIL import Image
import torch
import logging

# 设置日志记录
logging.basicConfig(level=logging.INFO)

# 检查是否有可用的GPU
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
logging.info(f"Using device: {device}")

# 加载处理器和模型，并将模型移到GPU
processor = AutoImageProcessor.from_pretrained("Falconsai/nsfw_image_detection")
model = AutoModelForImageClassification.from_pretrained("Falconsai/nsfw_image_detection").to(device)
logging.info(f"Model device: {next(model.parameters()).device}")

# 定义类别名称和NSFW阈值
labels = ["safe_for_work", "not_safe_for_work"]
nsfw_threshold = 0.3  # 设置阈值，超过该值认为是NSFW

# 指定目录路径
root_dir = r"E:\抖音\不是你的胖头鱼\新建文件夹"

# 支持的视频文件扩展名
video_extensions = ['.mp4', '.mov', '.avi', '.mkv']

# 跳过和处理的时间间隔（秒）
skip_seconds = 5  # 每次跳过1秒
frames_to_process = 1  # 每次处理1帧

def process_frames(frames, video_path, output_dir, base_frame_count):
    inputs = processor(images=frames, return_tensors="pt").to(device)
    with torch.no_grad():
        outputs = model(**inputs)
    logits = outputs.logits
    probabilities = torch.nn.functional.softmax(logits, dim=-1)
    nsfw_probs = probabilities[:, 1].cpu().numpy()  # 获取NSFW类别的概率

    for i, prob in enumerate(nsfw_probs):
        frame_id = base_frame_count + i
        logging.info(f"Video {video_path}, Frame {frame_id} - NSFW Probability: {prob}")
        if prob > nsfw_threshold:
            prob_str = f"{prob:.3f}".split('.')[1]  # 获取NSFW概率的小数部分
            screenshot_path = os.path.join(output_dir, f"{prob_str.zfill(3)}_{os.path.splitext(os.path.basename(video_path))[0]}_nsfw_frame_{frame_id}.jpg")
            try:
                frames[i].save(screenshot_path)
                logging.info(f"NSFW frame saved at: {screenshot_path}")
            except Exception as e:
                logging.error(f"Error saving frame {frame_id}: {e}")

def frame_extraction_worker(video_path, frame_queue, skip_frames, process_frames_count):
    cap = cv2.VideoCapture(video_path)
    frame_count = 0
    while cap.isOpened():
        # 读取 process_frames_count 的帧数
        frames = []
        for _ in range(process_frames_count):
            ret, frame = cap.read()
            if not ret:
                cap.release()
                frame_queue.put((None, None))  # Sentinel to signal end of video
                return
            frames.append(Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)))
            frame_count += 1
        if frames:
            frame_queue.put((frames, frame_count))
        
        # 跳过 skip_frames
        frame_count += skip_frames
        cap.set(cv2.CAP_PROP_POS_FRAMES, frame_count)
    cap.release()
    frame_queue.put((None, None))  # Sentinel to signal end of video

# 遍历指定目录及其子目录中的所有视频文件
for subdir, dirs, files in os.walk(root_dir):
    for file in files:
        if any(file.lower().endswith(ext) for ext in video_extensions):
            video_path = os.path.join(subdir, file)
            output_dir = subdir
            os.makedirs(output_dir, exist_ok=True)

            # 获取视频的帧率
            cap = cv2.VideoCapture(video_path)
            fps = cap.get(cv2.CAP_PROP_FPS)
            skip_frames = int(fps * skip_seconds)
            cap.release()

            frame_queue = Queue(maxsize=10)
            extraction_thread = threading.Thread(target=frame_extraction_worker, args=(video_path, frame_queue, skip_frames, frames_to_process))
            extraction_thread.start()

            while True:
                try:
                    frames, base_frame_count = frame_queue.get(timeout=30)  # 设定超时来防止阻塞
                    if frames is None:
                        break
                    process_frames(frames, video_path, output_dir, base_frame_count - len(frames))
                except Empty:
                    logging.warning("Queue is empty. Exiting.")
                    break

            extraction_thread.join()

cv2.destroyAllWindows()
