# Tree-of-Evolution 简化版（通义千问版）

这是 Tree-of-Evolution 项目的简化版本，使用通义千问（Qwen）API 替代 OpenAI API，目标生成 **500条** 高质量代码指令数据。

## 主要改动

1. **API 替换**：从 OpenAI API 改为通义千问 DashScope API
2. **简化流程**：减少轮数和每个样本的变体数，快速生成500条数据
3. **优化参数**：针对通义千问模型调整了默认参数

## 安装

### 1. 安装依赖

```bash
pip install -r requirements.txt
```

### 2. 配置 API Key

创建 `.env` 文件：

```bash
cp .env.example .env
```

编辑 `.env` 文件，填入你的 DashScope API Key：

```bash
DASHSCOPE_API_KEY=your_dashscope_api_key_here
```

**获取 API Key**：
1. 访问 [阿里云 DashScope 控制台](https://dashscope.console.aliyun.com/)
2. 注册/登录账号
3. 创建 API Key
4. 将 API Key 填入 `.env` 文件

## 快速开始

### 准备种子数据

创建 `data/seed/seed_samples.json` 文件，格式如下：

```json
[
  {
    "id": "1",
    "content": "def factorial(n):\n    if n <= 1:\n        return 1\n    return n * factorial(n-1)"
  },
  {
    "id": "2",
    "content": "def fibonacci(n):\n    if n <= 1:\n        return n\n    return fibonacci(n-1) + fibonacci(n-2)"
  }
]
```

**注意**：简化版建议使用 50 个种子样本，每个样本会生成 2 个变体，经过两轮演化后大约生成 500 条数据。

### 运行完整流程

**基本用法（使用默认参数，读取50个种子样本）**：

```bash
python run_pipeline.py
```

**自定义种子样本数量**：

```bash
# 读取前 100 个种子样本
python run_pipeline.py --seed_samples 100

# 读取全部种子样本
python run_pipeline.py --seed_samples 0

# 完整参数示例
python run_pipeline.py \
    --seed_samples 100 \
    --target_samples 1000 \
    --num_responses 3 \
    --model_name qwen-turbo
```

**参数说明**：
- `--seed_samples`: 从种子数据中读取的样本数量（默认: 50，设置为 0 表示读取全部）
- `--target_samples`: 目标生成的数据量（默认: 500）
- `--num_responses`: 每个样本生成的变体数（默认: 2）
- `--model_name`: 使用的模型名称（默认: qwen-plus）

这个脚本会自动执行：
1. 第1轮：从种子数据生成初始指令（直接合成）
2. 复杂度评分
3. 多样性评分
4. 第2轮：优化驱动演化
5. 复杂度评分
6. 多样性评分
7. 最终数据整理和过滤

### 手动运行各步骤

如果你想手动控制每个步骤：

#### 第1轮：指令合成

```bash
PYTHONPATH=. python src/instruction_synthesis.py \
    --input_path data/seed/seed_samples.json \
    --output_dir data/round1_synthesis \
    --model_name qwen-plus \
    --num_threads 4 \
    --num_responses 2 \
    --temperature 1.0 \
    --max_tokens 2048 \
    --max_samples 50
```

#### 复杂度评分

```bash
PYTHONPATH=. python src/complexity_scoring.py \
    --input_path data/round1_synthesis/all_questions_*.json \
    --output_dir data/round1_complexity \
    --model_name qwen-plus \
    --num_threads 4 \
    --temperature 0.0
```

#### 多样性评分

```bash
PYTHONPATH=. python src/diversity_scoring.py \
    --input_path data/round1_complexity/all_questions_*_w_complexity_scores.json \
    --output_path data/round1_diversity/questions_w_diversity_scores.json \
    --model_name "Alibaba-NLP/gte-large-en-v1.5" \
    --batch_size 10 \
    --device auto
```

#### 第2轮：优化驱动演化

```bash
PYTHONPATH=. python src/instruction_synthesis.py \
    --input_path data/round1_diversity/questions_w_diversity_scores.json \
    --output_dir data/round2_synthesis \
    --model_name qwen-plus \
    --num_threads 4 \
    --num_responses 2 \
    --temperature 1.0 \
    --max_tokens 2048 \
    --opt_evo
```

## 参数说明

### 模型选择

- `qwen-turbo`：速度更快，成本更低，适合快速测试
- `qwen-plus`：平衡速度和质量（推荐）
- `qwen-max`：质量最高，但速度较慢，成本较高

### 关键参数

- `--num_responses`：每个样本生成的变体数（默认：2，简化版）
- `--max_samples`：限制处理的样本数（用于控制生成数量）
- `--num_threads`：并行线程数（建议 4-8，根据API限流调整）
- `--temperature`：生成温度（合成用1.0，评分用0.0）

## 输出文件

最终结果保存在 `data/final/final_dataset_*.json`，包含以下字段：

```json
{
  "id": "1_0_1",
  "content": "编程问题内容...",
  "self complexity score": 7.5,
  "self diversity score": 0.8,
  "parent complexity score": 6.0,
  "parent diversity score": 0.7
}
```

## 成本估算

使用 `qwen-plus` 模型：
- 指令合成：约 0.012 元/1K tokens
- 复杂度评分：约 0.012 元/1K tokens
- 生成 500 条数据预计成本：约 10-30 元（取决于问题长度）

## 注意事项

1. **API 限流**：通义千问有 QPS 限制，建议 `num_threads` 设置为 4-8
2. **错误重试**：代码已实现自动重试机制，但遇到限流时会等待
3. **数据质量**：最终数据会根据复杂度（>=6.5）和多样性（>=0.6）进行过滤
4. **种子数据**：种子数据质量直接影响最终结果，建议使用高质量的代码片段

## 故障排除

### 1. API Key 错误

```
APIError: DASHSCOPE_API_KEY not found in environment variables
```

解决：检查 `.env` 文件是否正确配置，或设置环境变量：
```bash
export DASHSCOPE_API_KEY=your_key
```

### 2. 导入错误

```
ImportError: dashscope is not installed
```

解决：安装依赖
```bash
pip install -r requirements.txt
```

### 3. 限流错误

如果遇到 429 错误（限流），代码会自动重试并等待。可以：
- 减少 `--num_threads`
- 使用 `qwen-turbo` 模型（限流更宽松）

## 与原版对比

| 特性 | 原版 | 简化版 |
|------|------|--------|
| API | OpenAI | 通义千问 |
| 目标数据量 | 75k+ | 500 |
| 种子样本 | 5k | 50 |
| 每样本变体 | 3 | 2 |
| 演化轮数 | 多轮 | 2轮 |
| 成本 | 较高 | 较低 |

