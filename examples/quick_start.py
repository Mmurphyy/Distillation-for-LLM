#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
快速开始示例：演示如何使用知识蒸馏框架
"""

import os
import sys
import argparse
from pathlib import Path

# 添加项目根目录到系统路径
project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(project_root)

from src.utils import load_config, set_random_seed, get_device_info
from src.data import ExamDataset
from src.models import DistillationTrainer
from src.training import KnowledgeDistillationTrainer
from transformers import AutoTokenizer


def parse_args():
    """解析命令行参数"""
    parser = argparse.ArgumentParser(description="知识蒸馏框架快速开始示例")
    parser.add_argument(
        "--config", 
        type=str, 
        default=os.path.join(project_root, "configs/default_config.yaml"),
        help="配置文件路径",
    )
    parser.add_argument(
        "--data_file",
        type=str,
        default=os.path.join(project_root, "data/ARC-Challenge-Train.jsonl"),
        help="数据文件路径",
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        default=os.path.join(project_root, "outputs/quickstart"),
        help="输出目录",
    )
    parser.add_argument(
        "--epochs",
        type=int,
        default=1,
        help="训练轮数",
    )
    parser.add_argument(
        "--batch_size",
        type=int,
        default=4,
        help="训练批次大小",
    )
    return parser.parse_args()


def main():
    """主函数"""
    # 解析命令行参数
    args = parse_args()
    
    # 加载配置文件
    config = load_config(args.config)
    
    # 更新配置
    config["data"]["train_file"] = args.data_file
    config["data"]["batch_size"] = args.batch_size
    config["training"]["num_train_epochs"] = args.epochs
    config["output_dir"] = args.output_dir
    
    # 输出信息
    print("=" * 50)
    print("知识蒸馏框架快速开始示例")
    print("=" * 50)
    print(f"配置文件: {args.config}")
    print(f"数据文件: {args.data_file}")
    print(f"输出目录: {args.output_dir}")
    print(f"训练轮数: {args.epochs}")
    print(f"批次大小: {args.batch_size}")
    print("=" * 50)
    
    # 检查数据文件是否存在
    if not os.path.exists(args.data_file):
        print(f"错误: 数据文件不存在 {args.data_file}")
        return
    
    # 创建输出目录
    os.makedirs(args.output_dir, exist_ok=True)
    
    # 设置随机种子
    set_random_seed(config["random_seed"])
    print(f"已设置随机种子: {config['random_seed']}")
    
    # 获取设备信息
    device_info = get_device_info()
    print(f"设备信息: {'GPU' if device_info['cuda_available'] else 'CPU'}")
    if device_info["cuda_available"]:
        print(f"  GPU: {device_info['device_name']}")
        print(f"  显存: {device_info['memory_allocated']:.2f}MB / {device_info['memory_cached']:.2f}MB")
    
    # 加载分词器
    print("\n加载分词器...")
    try:
        teacher_tokenizer = AutoTokenizer.from_pretrained(
            config["model"]["teacher_model"],
            use_fast=True,
        )
        student_tokenizer = AutoTokenizer.from_pretrained(
            config["model"]["student_model"],
            use_fast=True,
        )
        print("分词器加载成功")
    except Exception as e:
        print(f"加载分词器时出错: {e}")
        return
    
    # 准备数据集
    print("\n准备数据集...")
    try:
        dataset = ExamDataset(config)
        dataloaders = dataset.create_dataloaders(
            tokenizer_teacher=teacher_tokenizer,
            tokenizer_student=student_tokenizer,
        )
        print(f"数据集加载成功")
    except Exception as e:
        print(f"准备数据集时出错: {e}")
        return
    
    # 创建蒸馏训练器
    print("\n创建蒸馏训练器...")
    try:
        print("注意: 首次加载大型教师模型可能需要一些时间...")
        distillation_trainer = DistillationTrainer(config)
        print("蒸馏训练器创建成功")
    except Exception as e:
        print(f"创建蒸馏训练器时出错: {e}")
        return
    
    # 创建知识蒸馏训练器
    print("\n创建知识蒸馏训练器...")
    try:
        trainer = KnowledgeDistillationTrainer(config, distillation_trainer, dataloaders)
        print("知识蒸馏训练器创建成功")
    except Exception as e:
        print(f"创建知识蒸馏训练器时出错: {e}")
        return
    
    # 开始训练
    print("\n开始训练...")
    try:
        metrics_history = trainer.train()
        print("训练完成!")
        
        # 输出最终评估指标
        if metrics_history:
            best_metric = max([m.get("accuracy", 0.0) for m in metrics_history])
            print(f"最佳准确率: {best_metric:.4f}")
        
        print(f"\n模型已保存到: {args.output_dir}")
    except Exception as e:
        print(f"训练过程中出错: {e}")
        return
    
    print("\n快速开始示例结束")
    print("接下来，您可以使用以下命令评估模型:")
    print(f"python scripts/evaluate.py --model_path {args.output_dir}/best_model --test_file {args.data_file}")
    print("\n或使用以下命令进行推理:")
    print(f"python scripts/inference.py --model_path {args.output_dir}/best_model --interactive")


if __name__ == "__main__":
    main() 
