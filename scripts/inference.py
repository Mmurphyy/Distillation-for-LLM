#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
推理脚本：用于使用知识蒸馏模型进行推理
支持交互式命令行和批处理文件两种模式
"""

import argparse
import json
import os
import sys
import time
from typing import Dict, Any, List, Optional

import torch
import jsonlines
from tqdm import tqdm
from transformers import AutoModelForSeq2SeqLM, AutoTokenizer

# 添加项目根目录到系统路径
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.utils import (
    load_config,
    get_logger,
    get_device_info,
    format_time,
)


def parse_args():
    """解析命令行参数"""
    parser = argparse.ArgumentParser(description="知识蒸馏模型推理脚本")
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
        help="模型路径",
    )
    parser.add_argument(
        "--input_file",
        type=str,
        default=None,
        help="输入文件路径，如果提供，将进行批处理推理",
    )
    parser.add_argument(
        "--output_file",
        type=str,
        default=None,
        help="输出文件路径，用于保存批处理推理结果",
    )
    parser.add_argument(
        "--batch_size",
        type=int,
        default=None,
        help="批处理大小，如果提供，将覆盖配置文件中的设置",
    )
    parser.add_argument(
        "--quantization",
        type=str,
        choices=["none", "int8", "int4"],
        default=None,
        help="量化模式，如果提供，将覆盖配置文件中的设置",
    )
    parser.add_argument(
        "--max_new_tokens",
        type=int,
        default=None,
        help="生成的最大token数，如果提供，将覆盖配置文件中的设置",
    )
    parser.add_argument(
        "--num_beams",
        type=int,
        default=None,
        help="束搜索的束宽，如果提供，将覆盖配置文件中的设置",
    )
    parser.add_argument(
        "--interactive",
        action="store_true",
        help="交互式模式",
    )
    return parser.parse_args()


def update_config_with_args(config: Dict[str, Any], args) -> Dict[str, Any]:
    """根据命令行参数更新配置"""
    # 更新推理配置
    if args.batch_size:
        config["inference"]["batch_size"] = args.batch_size
    if args.quantization:
        config["inference"]["quantization"] = args.quantization
    if args.max_new_tokens:
        config["evaluation"]["max_new_tokens"] = args.max_new_tokens
    if args.num_beams:
        config["evaluation"]["num_beams"] = args.num_beams
    
    return config


def load_model(model_path: str, config: Dict[str, Any]) -> tuple:
    """加载模型和分词器"""
    # 量化配置
    quantization = config["inference"]["quantization"]
    device_map = config["inference"]["device_map"]
    
    # 加载模型
    model_kwargs = {
        "device_map": device_map,
    }
    
    # 如果启用了量化
    if quantization == "int8":
        model_kwargs["load_in_8bit"] = True
    elif quantization == "int4":
        model_kwargs["load_in_4bit"] = True
        
    model = AutoModelForSeq2SeqLM.from_pretrained(model_path, **model_kwargs)
    tokenizer = AutoTokenizer.from_pretrained(model_path)
    
    return model, tokenizer


def format_question(stem: str, choices: List[Dict[str, str]]) -> str:
    """格式化问题"""
    options_text = ""
    for choice in choices:
        label = choice.get("label", "")
        text = choice.get("text", "")
        options_text += f"{label}. {text}\n"
    
    return f"问题：{stem}\n选项：\n{options_text.strip()}\n答案："


def batch_inference(
    model: AutoModelForSeq2SeqLM,
    tokenizer: AutoTokenizer,
    input_file: str,
    output_file: str,
    config: Dict[str, Any],
    logger: Optional[object] = None,
) -> None:
    """批处理推理"""
    if logger:
        logger.info(f"开始批处理推理: {input_file}")
    else:
        print(f"开始批处理推理: {input_file}")
    
    # 加载输入数据
    records = []
    with jsonlines.open(input_file) as reader:
        for obj in reader:
            try:
                question_data = obj
                # 处理ARC格式
                if isinstance(obj.get("question", None), str):
                    obj["question"] = json.loads(obj["question"])
                
                if "stem" not in obj and "question" in obj:
                    if isinstance(obj["question"], dict) and "stem" in obj["question"]:
                        question_data = obj["question"]
                        question_data["answerKey"] = obj.get("answerKey", "")
                        
                # 提取并标准化数据
                stem = question_data.get("stem", "")
                choices = question_data.get("choices", [])
                answer_key = question_data.get("answerKey", "")
                
                records.append({
                    "id": obj.get("id", ""),
                    "stem": stem,
                    "choices": choices,
                    "answer_key": answer_key
                })
            except Exception as e:
                if logger:
                    logger.error(f"处理记录时出错: {e}")
                else:
                    print(f"处理记录时出错: {e}")
                continue
    
    if not records:
        if logger:
            logger.error(f"输入文件中没有有效记录: {input_file}")
        else:
            print(f"输入文件中没有有效记录: {input_file}")
        return
    
    # 批处理参数
    batch_size = config["inference"]["batch_size"]
    max_new_tokens = config["evaluation"]["max_new_tokens"]
    num_beams = config["evaluation"]["num_beams"]
    
    # 准备输出
    results = []
    
    # 设置设备
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    # 开始处理
    total_time = 0.0
    for i in tqdm(range(0, len(records), batch_size), desc="批处理推理"):
        batch = records[i:i+batch_size]
        
        # 准备输入
        questions = [format_question(record["stem"], record["choices"]) for record in batch]
        
        # 编码输入
        inputs = tokenizer(
            questions,
            max_length=config["data"]["max_seq_length"],
            truncation=True,
            padding=True,
            return_tensors="pt",
        ).to(device)
        
        # 记录推理时间
        start_time = time.time()
        
        # 推理
        with torch.no_grad():
            outputs = model.generate(
                input_ids=inputs["input_ids"],
                attention_mask=inputs["attention_mask"],
                max_new_tokens=max_new_tokens,
                num_beams=num_beams,
                early_stopping=True,
            )
        
        # 计算推理时间
        inference_time = time.time() - start_time
        total_time += inference_time
        
        # 解码输出
        predicted_answers = tokenizer.batch_decode(outputs, skip_special_tokens=True)
        
        # 保存结果
        for j, record in enumerate(batch):
            result = {
                "id": record["id"],
                "question": record["stem"],
                "choices": record["choices"],
                "predicted_answer": predicted_answers[j].strip(),
                "reference_answer": record.get("answer_key", ""),
            }
            results.append(result)
    
    # 保存结果
    with jsonlines.open(output_file, "w") as writer:
        for result in results:
            writer.write(result)
    
    # 输出统计信息
    avg_time = total_time / len(records)
    if logger:
        logger.info(f"批处理推理完成! 共处理 {len(records)} 条记录")
        logger.info(f"平均推理时间: {avg_time:.4f}秒/样本")
        logger.info(f"结果已保存到: {output_file}")
    else:
        print(f"批处理推理完成! 共处理 {len(records)} 条记录")
        print(f"平均推理时间: {avg_time:.4f}秒/样本")
        print(f"结果已保存到: {output_file}")


def interactive_inference(
    model: AutoModelForSeq2SeqLM, 
    tokenizer: AutoTokenizer,
    config: Dict[str, Any],
    logger: Optional[object] = None,
) -> None:
    """交互式推理"""
    if logger:
        logger.info("启动交互式推理模式")
    
    print("=" * 50)
    print("知识蒸馏模型交互式推理模式")
    print("输入 'exit', 'quit', 或 'q' 退出")
    print("=" * 50)
    
    # 推理参数
    max_new_tokens = config["evaluation"]["max_new_tokens"]
    num_beams = config["evaluation"]["num_beams"]
    
    # 设置设备
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    while True:
        try:
            # 获取用户输入
            print("\n请输入问题（多行输入模式，输入空行结束）:")
            lines = []
            while True:
                line = input()
                if not line.strip():
                    break
                lines.append(line)
            
            question = "\n".join(lines)
            
            if question.lower() in ["exit", "quit", "q"]:
                print("退出交互式推理模式")
                break
            
            if not question.strip():
                continue
            
            # 编码输入
            inputs = tokenizer(
                question,
                max_length=config["data"]["max_seq_length"],
                truncation=True,
                padding=True,
                return_tensors="pt",
            ).to(device)
            
            # 记录推理时间
            start_time = time.time()
            
            # 推理
            with torch.no_grad():
                outputs = model.generate(
                    input_ids=inputs["input_ids"],
                    attention_mask=inputs["attention_mask"],
                    max_new_tokens=max_new_tokens,
                    num_beams=num_beams,
                    early_stopping=True,
                )
            
            # 计算推理时间
            inference_time = time.time() - start_time
            
            # 解码输出
            predicted_answer = tokenizer.batch_decode(outputs, skip_special_tokens=True)[0]
            
            # 打印结果
            print("\n回答:", predicted_answer.strip())
            print(f"推理时间: {inference_time:.4f}秒")
            
        except KeyboardInterrupt:
            print("\n退出交互式推理模式")
            break
        except Exception as e:
            error_msg = f"推理出错: {e}"
            if logger:
                logger.error(error_msg)
            print("\n" + error_msg)


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
        name="inference",
        log_file=os.path.join(log_dir, "inference.log"),
    )
    
    # 输出设备信息
    device_info = get_device_info()
    logger.info(f"设备信息: {device_info}")
    
    # 检查模型路径
    model_path = args.model_path
    if not os.path.exists(model_path):
        logger.error(f"模型路径不存在: {model_path}")
        sys.exit(1)
    
    # 加载模型
    logger.info(f"加载模型: {model_path}")
    model, tokenizer = load_model(model_path, config)
    logger.info("模型加载成功")
    
    # 执行推理
    if args.input_file:
        # 批处理模式
        if not args.output_file:
            # 如果没有指定输出文件，生成默认输出文件名
            output_dir = os.path.dirname(args.input_file)
            input_filename = os.path.basename(args.input_file)
            output_filename = f"results_{input_filename}"
            args.output_file = os.path.join(output_dir, output_filename)
        
        batch_inference(model, tokenizer, args.input_file, args.output_file, config, logger)
    elif args.interactive:
        # 交互式模式
        interactive_inference(model, tokenizer, config, logger)
    else:
        # 如果既没有指定输入文件也没有指定交互式模式，默认进入交互式模式
        logger.info("未指定输入文件，启动交互式模式")
        interactive_inference(model, tokenizer, config, logger)
    
    logger.info("推理过程结束")


if __name__ == "__main__":
    main() 