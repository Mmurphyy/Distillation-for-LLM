#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
训练脚本：用于训练知识蒸馏模型
"""

import argparse
import os
import sys
import time
from typing import Dict, Any

import torch
from transformers import AutoTokenizer

# 设置环境变量，避免tokenizers警告
os.environ["TOKENIZERS_PARALLELISM"] = "false"

# 添加项目根目录到系统路径
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.data import ExamDataset
from src.models import DistillationTrainer
from src.training import KnowledgeDistillationTrainer
from src.utils import (
    set_random_seed,
    load_config,
    save_config,
    get_logger,
    get_device_info,
    print_model_info,
    format_time,
)


def parse_args():
    """解析命令行参数"""
    parser = argparse.ArgumentParser(description="知识蒸馏模型训练脚本")
    parser.add_argument(
        "--config", 
        type=str, 
        default="configs/default_config.yaml",
        help="配置文件路径",
    )
    parser.add_argument(
        "--train_file",
        type=str,
        default=None,
        help="训练数据文件路径，如果提供，将覆盖配置文件中的设置",
    )
    parser.add_argument(
        "--valid_file",
        type=str,
        default=None,
        help="验证数据文件路径，如果提供，将覆盖配置文件中的设置",
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        default=None,
        help="输出目录，如果提供，将覆盖配置文件中的设置",
    )
    parser.add_argument(
        "--teacher_model",
        type=str,
        default="/home/zp/.cache/modelscope/hub/models/Qwen/Qwen3-4B",
        help="教师模型路径或名称，如果提供，将覆盖配置文件中的设置",
    )
    parser.add_argument(
        "--student_model",
        type=str,
        default="/home/zp/.cache/modelscope/hub/models/Qwen/Qwen3-0.6B",
        help="学生模型路径或名称，如果提供，将覆盖配置文件中的设置",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=None,
        help="随机种子，如果提供，将覆盖配置文件中的设置",
    )
    parser.add_argument(
        "--fp16",
        action="store_true",
        help="是否使用混合精度训练",
    )
    parser.add_argument(
        "--bf16",
        action="store_true",
        help="是否使用BF16精度训练",
    )
    parser.add_argument(
        "--epochs",
        type=int,
        default=None,
        help="训练轮数，如果提供，将覆盖配置文件中的设置",
    )
    parser.add_argument(
        "--batch_size",
        type=int,
        default=None,
        help="训练批次大小，如果提供，将覆盖配置文件中的设置",
    )
    return parser.parse_args()


def update_config_with_args(config: Dict[str, Any], args) -> Dict[str, Any]:
    """根据命令行参数更新配置"""
    # 更新数据文件路径
    if args.train_file:
        config["data"]["train_file"] = args.train_file
    if args.valid_file:
        config["data"]["valid_file"] = args.valid_file
    
    # 更新模型配置
    if args.teacher_model:
        config["model"]["teacher_model"] = args.teacher_model
    if args.student_model:
        config["model"]["student_model"] = args.student_model
    
    # 更新输出目录
    if args.output_dir:
        config["output_dir"] = args.output_dir
    
    # 更新随机种子
    if args.seed:
        config["random_seed"] = args.seed
    
    # 更新训练配置
    if args.fp16:
        config["training"]["fp16"] = True
        config["training"]["bf16"] = False
    if args.bf16:
        config["training"]["bf16"] = True
        config["training"]["fp16"] = False
    if args.epochs:
        config["training"]["num_train_epochs"] = args.epochs
    if args.batch_size:
        config["data"]["batch_size"] = args.batch_size
    
    return config


def main():
    """主函数"""
    # 解析命令行参数
    args = parse_args()
    
    # 加载配置文件
    config = load_config(args.config)
    
    # 根据命令行参数更新配置
    config = update_config_with_args(config, args)
    
    # 创建输出目录
    output_dir = config["output_dir"]
    os.makedirs(output_dir, exist_ok=True)
    
    # 保存配置到输出目录
    save_config(config, os.path.join(output_dir, "config.yaml"))
    
    # 设置随机种子
    set_random_seed(config["random_seed"])
    
    # 创建日志记录器
    logger = get_logger(
        name="train",
        log_file=os.path.join(config["log_dir"], "train.log"),
    )
    
    # 输出设备信息
    device_info = get_device_info()
    logger.info(f"设备信息: {device_info}")
    
    # 加载分词器
    logger.info("加载分词器...")
    teacher_tokenizer = AutoTokenizer.from_pretrained(
        config["model"]["teacher_model"],
        use_fast=True,
        torch_dtype=torch.float16,    # 改用 fp16
        quantization_config=None      # 或者不传 FP8 配置
    )
    student_tokenizer = AutoTokenizer.from_pretrained(
        config["model"]["student_model"],
        use_fast=True,
        torch_dtype=torch.float16,    # 改用 fp16
        quantization_config=None      # 或者不传 FP8 配置
    )
    
    # 准备数据集
    logger.info("准备数据集...")
    dataset = ExamDataset(config)
    dataloaders = dataset.create_dataloaders(
        tokenizer_teacher=teacher_tokenizer,
        tokenizer_student=student_tokenizer,
    )
    
    # 创建教师-学生蒸馏训练器
    logger.info("创建蒸馏训练器...")
    distillation_trainer = DistillationTrainer(config)
    
    # 打印模型信息
    logger.info("教师模型信息:")
    print_model_info(distillation_trainer.teacher_model)
    logger.info("学生模型信息:")
    print_model_info(distillation_trainer.student.student_model)
    
    # 创建知识蒸馏训练器
    trainer = KnowledgeDistillationTrainer(config, distillation_trainer, dataloaders)
    
    # 开始训练
    logger.info("开始训练...")
    start_time = time.time()
    
    try:
        metrics_history = trainer.train()
        
        # 训练完成，计算总用时
        total_time = time.time() - start_time
        formatted_time = format_time(total_time)
        logger.info(f"训练完成！总用时: {formatted_time}")
        
        # 输出最终评估指标
        best_metric = max([m.get("accuracy", 0.0) for m in metrics_history]) if metrics_history else 0.0
        logger.info(f"最佳准确率: {best_metric:.4f}")
        
    except KeyboardInterrupt:
        logger.info("训练被中断")
        # 保存当前模型
        trainer._save_checkpoint("interrupted")
        logger.info(f"已保存中断时的模型到 {os.path.join(output_dir, 'interrupted')}")
    except Exception as e:
        logger.error(f"训练发生错误: {e}")
        raise
    
    logger.info(f"模型已保存到 {output_dir}")
    logger.info("训练过程结束")


if __name__ == "__main__":
    main() 