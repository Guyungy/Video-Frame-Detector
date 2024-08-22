该脚本遍历指定目录及其子目录中的所有视频文件，检测视频中的内容，并保存截图

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

3. **运行脚本**，它将遍历指定目录及其子目录中的所有视频文件，并进行内容检测。如果检测到内容，将保存相应的截图到与视频文件相同的目录中。


### GPU运行

1. 卸载当前的PyTorch

``` bash
pip uninstall torch torchvision torchaudio
``` 

2. 根据你的CUDA版本，安装正确的PyTorch版本。你可以从PyTorch官网的安装页面获取适合你的CUDA版本的安装命令。例如，如果你使用CUDA 11.8，可以使用以下命令：

``` bash
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118
```

3. 安装完成后，运行以下代码来检查CUDA是否可用：

``` python
import torch
print(torch.cuda.is_available())
print(torch.cuda.device_count())
print(torch.cuda.current_device())
print(torch.cuda.get_device_name(0))
```

如果CUDA配置正确，这些命令应该输出True、设备数量、当前设备索引和设备名称。

## 注意事项

- 确保在运行脚本前，指定的目录路径是正确的，并且程序具有在该目录中创建文件的权限。
- 根据需要调整检测间隔（例如，每30帧检测一次）以平衡检测频率和性能。
- 确保安装了所有必要的库和依赖项。

通过这些步骤，你可以使用该脚本在指定目录及其子目录中检测视频文件的内容，并保存相关的截图。
