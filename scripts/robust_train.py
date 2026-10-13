#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
稳健版蒸馏训练脚本：简化数据处理，专注于稳定训练
"""

import argparse
import json
import os
import random
from typing import Dict, List

import jsonlines
import pandas as pd
import torch
import torch.nn.functional as F
from datasets import Dataset
from peft import LoraConfig, TaskType, get_peft_model
from torch.utils.data import DataLoader
from tqdm.auto import tqdm
from transformers import (
    AutoTokenizer,
    AutoModelForCausalLM,
    get_cosine_schedule_with_warmup,
)

# 设置环境变量，避免tokenizers警告
os.environ["TOKENIZERS_PARALLELISM"] = "false"

def parse_args():
    parser = argparse.ArgumentParser(description="稳健版知识蒸馏训练脚本")
    parser.add_argument(
        "--train_file",
        type=str,
        default="data/ARC-Challenge-Train.jsonl",
        help="训练数据文件路径",
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        default="outputs/robust",
        help="输出目录",
    )
    parser.add_argument(
        "--teacher_model",
        type=str,
        default="/home/zp/.cache/modelscope/hub/models/Qwen/Qwen3-4B",
        help="教师模型路径",
    )
    parser.add_argument(
        "--student_model",
        type=str,
        default="/home/zp/.cache/modelscope/hub/models/Qwen/Qwen3-0.6B",
        help="学生模型路径",
    )
    parser.add_argument(
        "--max_samples",
        type=int,
        default=100,
        help="最大训练样本数",
    )
    parser.add_argument(
        "--epochs",
        type=int,
        default=1,
        help="训练轮数",
    )
    parser.add_argument(
        "--lr",
        type=float,
        default=1e-4,
        help="学习率",
    )
    parser.add_argument(
        "--max_seq_length",
        type=int,
        default=128,
        help="最大序列长度",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="随机种子",
    )
    parser.add_argument(
        "--log_steps",
        type=int,
        default=10,
        help="日志记录步数",
    )
    parser.add_argument(
        "--save_steps",
        type=int,
        default=50,
        help="保存步数",
    )
    return parser.parse_args()

def set_random_seed(seed):
    """设置随机种子"""
    random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

def load_jsonl_samples(file_path: str, max_samples: int) -> list:
    """加载JSONL格式的考试题目数据，并限制样本数量"""
    if not os.path.exists(file_path):
        raise FileNotFoundError(f"数据文件不存在: {file_path}")
    
    samples = []
    with jsonlines.open(file_path) as reader:
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
                
                options_text = ""
                for choice in choices:
                    label = choice.get("label", "")
                    text = choice.get("text", "")
                    options_text += f"{label}. {text}\n"
                
                prompt = f"问题：{stem}\n选项：\n{options_text.strip()}\n答案："
                target = answer_key
                
                samples.append((prompt, target))
                
                if len(samples) >= max_samples:
                    break
            except Exception as e:
                print(f"处理记录时出错: {e}")
                continue
    
    return samples

def main():
    """主函数"""
    args = parse_args()
    set_random_seed(args.seed)
    
    # 创建输出目录
    os.makedirs(args.output_dir, exist_ok=True)
    
    # 设置设备
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"使用设备: {device}")
    
    # 加载模型和分词器
    print("加载教师模型和分词器...")
    teacher_tokenizer = AutoTokenizer.from_pretrained(args.teacher_model)
    teacher_model = AutoModelForCausalLM.from_pretrained(
        args.teacher_model,
        device_map="auto",
        torch_dtype=torch.float16,
    )
    
    # 冻结教师模型参数
    for param in teacher_model.parameters():
        param.requires_grad = False
    
    print("加载学生模型和分词器...")
    student_tokenizer = AutoTokenizer.from_pretrained(args.student_model)
    student_model = AutoModelForCausalLM.from_pretrained(
        args.student_model,
        device_map="auto",
        torch_dtype=torch.float16,
    )
    
    # 为学生模型添加LoRA
    print("应用LoRA配置...")
    peft_config = LoraConfig(
        task_type=TaskType.CAUSAL_LM,
        inference_mode=False,
        r=8,
        lora_alpha=16,
        lora_dropout=0.05,
        target_modules=["q_proj", "k_proj", "v_proj", "o_proj"]
    )
    student_model = get_peft_model(student_model, peft_config)
    
    # 打印可训练参数信息
    trainable_params = 0
    all_params = 0
    for _, param in student_model.named_parameters():
        all_params += param.numel()
        if param.requires_grad:
            trainable_params += param.numel()
    print(f"学生模型可训练参数: {trainable_params:,d} ({100 * trainable_params / all_params:.2f}%)")
    
    # 加载数据
    print(f"加载训练数据，最多 {args.max_samples} 个样本...")
    train_samples = load_jsonl_samples(args.train_file, args.max_samples)
    print(f"已加载 {len(train_samples)} 个训练样本")
    print(train_samples[0])
    print(train_samples[1])
    print(train_samples[2])
    print(train_samples[3])
    print(train_samples[4])
    print(train_samples[5])
    print(train_samples[6])
    print(train_samples[7])
    print(train_samples[8])
    

    # 创建优化器
    optimizer = torch.optim.AdamW(
        [p for p in student_model.parameters() if p.requires_grad],
        lr=args.lr,
        weight_decay=0.01,
    )
    
    # 训练参数
    temperature = 2.0
    distillation_weight = 0.5
    ce_weight = 0.5
    
    # 训练循环
    global_step = 0
    
    print(f"开始训练，共 {args.epochs} 轮...")
    for epoch in range(args.epochs):
        print(f"开始第 {epoch + 1}/{args.epochs} 轮训练")
        student_model.train()
        
        # 打乱训练样本
        random.shuffle(train_samples)
        
        epoch_iterator = tqdm(train_samples, desc=f"训练 (Epoch {epoch + 1})")
        total_loss = 0
        
        for step, (prompt, target) in enumerate(epoch_iterator):
            # 1. 编码prompt（到“答案：”为止）
            inputs = teacher_tokenizer(
                prompt, 
                return_tensors="pt",
                truncation=True,
                max_length=args.max_seq_length,
                padding="max_length",
            ).to(device)
            
            # 2. 编码label（答案label，如'B'）
            label_inputs = teacher_tokenizer(
                target,
                return_tensors="pt",
                truncation=True,
                max_length=4,  # 答案通常很短
                padding="max_length"
            ).to(device)
            
            # 3. 构造labels，只对答案部分做监督
            labels = torch.full(inputs["input_ids"].shape, -100, dtype=torch.long, device=device)
            labels[:, -label_inputs["input_ids"].shape[1]:] = label_inputs["input_ids"]
            inputs["labels"] = labels
            
            # 4. 获取教师模型输出
            with torch.no_grad():
                teacher_outputs = teacher_model(**inputs)
            
            # 5. 获取学生模型输出
            student_outputs = student_model(**inputs)
            
            # 6. 交叉熵损失
            ce_loss = student_outputs.loss
            
            # 7. 蒸馏损失（只对答案token做KL散度）
            answer_start = inputs["input_ids"].shape[1] - label_inputs["input_ids"].shape[1]
            student_logits_ans = student_outputs.logits[:, answer_start:, :]
            teacher_logits_ans = teacher_outputs.logits[:, answer_start:, :]
            
            distillation_loss = F.kl_div(
                F.log_softmax(student_logits_ans / temperature, dim=-1),
                F.softmax(teacher_logits_ans / temperature, dim=-1),
                reduction="batchmean"
            ) * (temperature ** 2)
            
            # 8. 总损失
            loss = ce_weight * ce_loss + distillation_weight * distillation_loss
            
            # 9. 反向传播和优化
            loss.backward()
            torch.nn.utils.clip_grad_norm_(
                [p for p in student_model.parameters() if p.requires_grad],
                max_norm=1.0,
            )
            optimizer.step()
            optimizer.zero_grad()
            
            # 10. 日志与保存逻辑
            epoch_iterator.set_postfix({
                "loss": f"{loss.item():.4f}",
                "ce_loss": f"{ce_loss.item():.4f}",
                "kd_loss": f"{distillation_loss.item():.4f}",
            })
            total_loss += loss.item()
            global_step += 1
            
            if global_step % args.log_steps == 0:
                print(f"步骤 {global_step} - 损失: {loss.item():.4f}, "
                      f"CE损失: {ce_loss.item():.4f}, "
                      f"KD损失: {distillation_loss.item():.4f}")
                # 生成时只用prompt到“答案：”为止
                gen_inputs = teacher_tokenizer(prompt, return_tensors="pt").to(device)
                with torch.no_grad():
                    outputs = student_model.generate(
                        input_ids=gen_inputs["input_ids"],
                        max_new_tokens=10,
                        num_beams=2,
                        do_sample=False,
                    )
                    gen_text = student_tokenizer.decode(outputs[0], skip_special_tokens=True)
                    print(f"输入: {prompt}")
                    print(f"目标: {target}")
                    print(f"生成: {gen_text}")
                    print("-" * 40)
            if global_step % args.save_steps == 0:
                save_path = os.path.join(args.output_dir, f"step_{global_step}")
                print(f"保存检查点到 {save_path}")
                student_model.save_pretrained(save_path)
                student_tokenizer.save_pretrained(save_path)
        
        # Epoch 完成
        avg_loss = total_loss / len(train_samples)
        print(f"Epoch {epoch + 1} 完成 - 平均损失: {avg_loss:.4f}")
        
        # 每个epoch结束后保存
        save_path = os.path.join(args.output_dir, f"epoch_{epoch + 1}")
        print(f"保存epoch检查点到 {save_path}")
        student_model.save_pretrained(save_path)
        student_tokenizer.save_pretrained(save_path)
    
    # 保存最终模型
    final_path = os.path.join(args.output_dir, "final_model")
    print(f"保存最终模型到 {final_path}")
    student_model.save_pretrained(final_path)
    student_tokenizer.save_pretrained(final_path)
    
    print("训练完成！")
    
    # 测试最终模型
    print("使用最终模型进行测试...")
    test_prompt = "问题：地球是太阳系中第几颗行星？\n选项：\nA. 第一颗\nB. 第三颗\nC. 第五颗\nD. 第八颗\n答案："
    
    test_inputs = teacher_tokenizer(test_prompt, return_tensors="pt").to(device)
    
    with torch.no_grad():
        outputs = student_model.generate(
            input_ids=test_inputs.input_ids,
            max_new_tokens=10,
            num_beams=2,
            do_sample=False,
        )
        
        gen_text = student_tokenizer.decode(outputs[0], skip_special_tokens=True)
        print(f"测试输入: {test_prompt}")
        print(f"模型输出: {gen_text}")

if __name__ == "__main__":
    main() 