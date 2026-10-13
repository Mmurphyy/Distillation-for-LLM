import os
import yaml
import random
import logging
from typing import Dict, Any, Union, Optional

import numpy as np
import torch
from transformers import set_seed


def set_random_seed(seed: int = 42):
    """
    设置随机种子，确保结果可复现
    """
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    set_seed(seed)


def load_config(config_path: str) -> Dict[str, Any]:
    """
    加载YAML配置文件
    """
    if not os.path.exists(config_path):
        raise FileNotFoundError(f"配置文件不存在: {config_path}")
    
    with open(config_path, "r", encoding="utf-8") as f:
        config = yaml.safe_load(f)
    
    return config


def save_config(config: Dict[str, Any], output_path: str):
    """
    保存配置到YAML文件
    """
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    
    with open(output_path, "w", encoding="utf-8") as f:
        yaml.dump(config, f, default_flow_style=False, allow_unicode=True)


def get_logger(name: str, log_file: Optional[str] = None, level: int = logging.INFO) -> logging.Logger:
    """
    创建日志记录器
    """
    logger = logging.getLogger(name)
    logger.setLevel(level)
    
    # 创建控制台处理器
    console_handler = logging.StreamHandler()
    console_handler.setLevel(level)
    
    # 定义日志格式
    formatter = logging.Formatter("%(asctime)s - %(name)s - %(levelname)s - %(message)s")
    console_handler.setFormatter(formatter)
    
    logger.addHandler(console_handler)
    
    # 如果提供了日志文件路径，添加文件处理器
    if log_file:
        os.makedirs(os.path.dirname(log_file), exist_ok=True)
        file_handler = logging.FileHandler(log_file)
        file_handler.setLevel(level)
        file_handler.setFormatter(formatter)
        logger.addHandler(file_handler)
    
    return logger


def get_device_info() -> Dict[str, Any]:
    """
    获取GPU/CPU信息
    """
    device_info = {
        "device": "cuda" if torch.cuda.is_available() else "cpu",
        "cuda_available": torch.cuda.is_available(),
    }
    
    if torch.cuda.is_available():
        device_info.update({
            "gpu_count": torch.cuda.device_count(),
            "current_device": torch.cuda.current_device(),
            "device_name": torch.cuda.get_device_name(0),
            "memory_allocated": torch.cuda.memory_allocated(0) / 1024**2,  # MB
            "memory_cached": torch.cuda.memory_reserved(0) / 1024**2,  # MB
        })
    
    return device_info


def print_model_info(model: torch.nn.Module) -> Dict[str, Any]:
    """
    打印模型信息，包括参数数量
    """
    # 计算参数总数
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    
    model_info = {
        "total_parameters": total_params,
        "trainable_parameters": trainable_params,
        "non_trainable_parameters": total_params - trainable_params,
    }
    
    # 打印信息
    print(f"模型总参数: {total_params:,}")
    print(f"可训练参数: {trainable_params:,}")
    print(f"冻结参数: {total_params - trainable_params:,}")
    
    return model_info


def format_time(seconds: float) -> str:
    """
    将秒数格式化为易读的时间字符串
    """
    hours, remainder = divmod(seconds, 3600)
    minutes, seconds = divmod(remainder, 60)
    
    if hours > 0:
        return f"{int(hours)}小时 {int(minutes)}分钟 {seconds:.2f}秒"
    elif minutes > 0:
        return f"{int(minutes)}分钟 {seconds:.2f}秒"
    else:
        return f"{seconds:.2f}秒" 
