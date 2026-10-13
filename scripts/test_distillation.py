#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
测试脚本：测试简单的知识蒸馏过程
"""

import os
import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import AutoTokenizer, AutoModelForCausalLM
from peft import LoraConfig, TaskType, get_peft_model

# 设置环境变量，避免tokenizers警告
os.environ["TOKENIZERS_PARALLELISM"] = "false"

def main():
    print("开始蒸馏测试...")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"使用设备: {device}")
    
    # 加载教师模型
    print("加载教师模型...")
    teacher_model_path = "/home/zp/.cache/modelscope/hub/models/Qwen/Qwen3-4B"
    teacher_tokenizer = AutoTokenizer.from_pretrained(teacher_model_path)
    teacher_model = AutoModelForCausalLM.from_pretrained(
        teacher_model_path,
        device_map="auto",
        torch_dtype=torch.float16,
    )
    
    # 冻结教师模型参数
    for param in teacher_model.parameters():
        param.requires_grad = False
    
    # 加载学生模型
    print("加载学生模型...")
    student_model_path = "/home/zp/.cache/modelscope/hub/models/Qwen/Qwen3-0.6B"
    student_tokenizer = AutoTokenizer.from_pretrained(student_model_path)
    student_model = AutoModelForCausalLM.from_pretrained(
        student_model_path,
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
    print(
        f"学生模型可训练参数: {trainable_params:,d} ({100 * trainable_params / all_params:.2f}%)"
    )
    
    # 准备输入示例
    print("准备测试样例...")
    prompt = "问题：地球是太阳系中第几颗行星？\n选项：\nA. 第一颗\nB. 第三颗\nC. 第五颗\nD. 第八颗\n答案："
    inputs = teacher_tokenizer(prompt, return_tensors="pt").to(device)
    
    # 获取教师模型输出
    print("获取教师模型输出...")
    with torch.no_grad():
        teacher_outputs = teacher_model(**inputs)
    
    # 获取学生模型输出
    print("获取学生模型输出...")
    student_outputs = student_model(**inputs)
    
    # 计算知识蒸馏损失
    print("计算蒸馏损失...")
    temperature = 2.0
    
    # 应用温度缩放
    teacher_logits = teacher_outputs.logits / temperature
    student_logits = student_outputs.logits / temperature
    
    # 计算KL散度损失
    loss = F.kl_div(
        F.log_softmax(student_logits, dim=-1),
        F.softmax(teacher_logits, dim=-1),
        reduction="batchmean",
    ) * (temperature ** 2)
    
    print(f"蒸馏损失: {loss.item()}")
    
    # 模拟反向传播
    print("执行反向传播...")
    loss.backward()
    
    # 创建优化器
    optimizer = torch.optim.AdamW(
        [p for p in student_model.parameters() if p.requires_grad], 
        lr=1e-4
    )
    
    # 更新参数
    optimizer.step()
    optimizer.zero_grad()
    
    print("蒸馏测试完成！")
    
    # 使用更新后的模型生成文本
    print("使用更新后的学生模型生成文本...")
    with torch.no_grad():
        outputs = student_model.generate(
            inputs.input_ids,
            max_new_tokens=10,
            do_sample=False,
            num_beams=2,
        )
    
    # 解码输出
    result = student_tokenizer.decode(outputs[0], skip_special_tokens=True)
    print(f"输入: {prompt}")
    print(f"学生模型输出: {result}")

if __name__ == "__main__":
    main() 