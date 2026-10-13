#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
评估脚本：用于评估知识蒸馏模型的性能
"""

import argparse
import os
import sys
import time
from typing import Dict, Any

import torch

# 添加项目根目录到系统路径
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.evaluation import Evaluator
from src.utils import (
    load_config,
    get_logger,
    get_device_info,
    format_time,
)


def parse_args():
    """解析命令行参数"""
    parser = argparse.ArgumentParser(description="知识蒸馏模型评估脚本")
    parser.add_argument(
        "--config", 
        type=str, 
        default="configs/default_config.yaml",
        help="配置文件路径",
    )
    parser.add_argument(
        "--model_path",
        type=str,
        required=True,
        help="待评估模型路径",
    )
    parser.add_argument(
        "--test_file",
        type=str,
        default=None,
        help="测试数据文件路径，如果提供，将覆盖配置文件中的设置",
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        default=None,
        help="评估结果输出目录，如果提供，将覆盖配置文件中的设置",
    )
    parser.add_argument(
        "--generate_explanations",
        action="store_true",
        help="是否生成解释",
    )
    parser.add_argument(
        "--batch_size",
        type=int,
        default=None,
        help="推理批次大小，如果提供，将覆盖配置文件中的设置",
    )
    parser.add_argument(
        "--quantization",
        type=str,
        choices=["none", "int8", "int4"],
        default=None,
        help="量化模式，如果提供，将覆盖配置文件中的设置",
    )
    return parser.parse_args()


def update_config_with_args(config: Dict[str, Any], args) -> Dict[str, Any]:
    """根据命令行参数更新配置"""
    # 更新数据文件路径
    if args.test_file:
        config["data"]["test_file"] = args.test_file
    
    # 更新输出目录
    if args.output_dir:
        config["log_dir"] = args.output_dir
    
    # 更新评估配置
    if args.generate_explanations:
        config["evaluation"]["generate_explanations"] = True
    
    # 更新推理配置
    if args.batch_size:
        config["inference"]["batch_size"] = args.batch_size
    if args.quantization:
        config["inference"]["quantization"] = args.quantization
    
    return config


def main():
    """主函数"""
    # 解析命令行参数
    args = parse_args()
    
    # 加载配置文件
    config = load_config(args.config)
    
    # 根据命令行参数更新配置
    config = update_config_with_args(config, args)
    
    # 创建日志目录
    log_dir = config["log_dir"]
    os.makedirs(log_dir, exist_ok=True)
    
    # 创建日志记录器
    logger = get_logger(
        name="evaluate",
        log_file=os.path.join(log_dir, "evaluate.log"),
    )
    
    # 输出设备信息
    device_info = get_device_info()
    logger.info(f"设备信息: {device_info}")
    
    # 检查模型路径
    model_path = args.model_path
    if not os.path.exists(model_path):
        logger.error(f"模型路径不存在: {model_path}")
        sys.exit(1)
    
    # 检查测试文件
    test_file = config["data"]["test_file"]
    if not os.path.exists(test_file):
        logger.error(f"测试文件不存在: {test_file}")
        sys.exit(1)
    
    # 创建评估器
    logger.info("创建评估器...")
    evaluator = Evaluator(config)
    
    # 开始评估
    logger.info(f"开始评估模型: {model_path}")
    logger.info(f"测试数据: {test_file}")
    start_time = time.time()
    
    try:
        metrics = evaluator.evaluate(model_path, test_file)
        
        # 评估完成，计算总用时
        total_time = time.time() - start_time
        formatted_time = format_time(total_time)
        logger.info(f"评估完成！总用时: {formatted_time}")
        
        # 输出评估指标
        logger.info("评估指标:")
        logger.info(f"  样本数量: {metrics['total_samples']}")
        logger.info(f"  正确数量: {metrics['correct_samples']}")
        logger.info(f"  准确率: {metrics['accuracy']:.4f}")
        logger.info(f"  平均推理时间: {metrics['avg_inference_time']:.4f}秒/样本")
        
        if "f1" in metrics:
            logger.info(f"  F1分数: {metrics['f1']:.4f}")
        if "exact_match" in metrics:
            logger.info(f"  精确匹配率: {metrics['exact_match']:.4f}")
        
    except Exception as e:
        logger.error(f"评估发生错误: {e}")
        raise
    
    logger.info("评估过程结束")


if __name__ == "__main__":
    main() 