# 基于大语言模型知识推理能力的蒸馏框架

## 项目简介
本项目实现了一个基于大语言模型知识推理能力的蒸馏框架，旨在将中文大模型的知识推理能力高效迁移到考试领域，尤其针对在有限计算资源下的应用场景。

## 核心特性
- **分层知识蒸馏**：题目理解层 + 知识推理层的双层架构
- **创新技术**：注意力增强机制、渐进式参数解冻、多任务学习框架、动态知识路由
- **工程优化**：混合精度训练、梯度累积、模型量化、推理加速
- **多域适用**：支持扩展到物理、化学、法律、医学等多个领域

## 环境配置
```bash
pip install -r requirements.txt
```

## 数据准备
将待处理的数据放入 `data` 目录下：
- 考试题目数据集：ARC-Challenge-Train.jsonl 等
- 对话数据集：可按照 src/data 模块中的格式进行准备

## 模型训练
```bash
# 使用默认配置训练
python scripts/train.py --config configs/default_config.yaml

# 自定义配置训练
python scripts/train.py --config configs/your_config.yaml
```

## 模型评估
```bash
python scripts/evaluate.py --model_path outputs/your_model --test_file data/your_test_file.jsonl
```

## 推理服务
```bash
python scripts/inference.py --model_path outputs/your_model
```

## 项目结构
```
蒸馏/
├── configs/             # 配置文件目录
├── data/                # 数据目录
├── outputs/             # 模型输出目录
├── scripts/             # 训练、评估、推理脚本
├── src/                 # 源代码
│   ├── data/            # 数据处理模块
│   ├── models/          # 模型定义模块
│   ├── training/        # 训练逻辑模块
│   ├── evaluation/      # 评估模块
│   └── utils/           # 工具函数模块
└── README.md            # 项目说明
```

## 性能指标
- **答案准确率**：模型在测试集上的正确答案比例
- **推理时间**：单题推理耗时
- **资源占用**：GPU显存使用量

## 适用场景
- T4 GPU等受限硬件环境
- 多领域知识推理任务
- 考试辅助系统 