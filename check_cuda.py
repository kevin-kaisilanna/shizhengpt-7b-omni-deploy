import torch  # 导入PyTorch
print(torch.cuda.is_available())  # 检查CUDA是否可用
print(torch.cuda.device_count())  # 检查可用的GPU数量
print(torch.cuda.get_device_name(0))  # 查看第1块GPU的型号（单卡就是0）