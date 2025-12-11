# 快速开始指南

## 5分钟快速上手

### 1. 安装依赖

```bash
pip install -r requirements.txt
```

### 2. 配置 API Key

```bash
# 复制环境配置文件
cp env.example .env

# 编辑 .env 文件，填入你的 DashScope API Key
# 获取地址: https://dashscope.console.aliyun.com/
```

### 3. 创建种子数据（可选）

如果你没有种子数据，可以使用提供的脚本创建示例数据：

```bash
python create_seed_data.py
```

这会创建 50 个示例 Python 代码片段作为种子数据。

### 4. 运行完整流程

```bash
python run_pipeline.py
```

脚本会自动执行所有步骤，最终生成约 500 条高质量数据。

## 预期时间

- 使用 `qwen-plus` 模型：约 1-2 小时
- 使用 `qwen-turbo` 模型：约 30-60 分钟

## 输出位置

最终结果保存在：
```
data/final/final_dataset_*.json
```

## 常见问题

### Q: API Key 在哪里获取？
A: 访问 [阿里云 DashScope 控制台](https://dashscope.console.aliyun.com/)，注册登录后创建 API Key。

### Q: 如何修改生成的数据量？
A: 编辑 `run_pipeline.py`，修改 `TARGET_SAMPLES` 变量。

### Q: 如何加快速度？
A: 
1. 使用 `qwen-turbo` 模型（修改 `MODEL_NAME`）
2. 减少 `NUM_RESPONSES`（每个样本的变体数）
3. 减少 `SEED_SAMPLES`（种子样本数）

### Q: 遇到限流错误怎么办？
A: 
1. 减少 `--num_threads` 参数（默认是4）
2. 等待一段时间后重试
3. 代码会自动重试，但限流时会等待更长时间

## 下一步

查看 `README.md` 了解更详细的配置和参数说明。


# 使用说明：设置种子数据读取数量

## 方法一：使用命令行参数（推荐）

在运行 `run_pipeline.py` 时，使用 `--seed_samples` 参数：

```bash
# 读取前 100 个种子样本
python run_pipeline.py --seed_samples 100

# 读取前 200 个种子样本
python run_pipeline.py --seed_samples 200

# 读取全部种子样本（设置为 0）
python run_pipeline.py --seed_samples 0
```

### 其他有用的参数

```bash
# 完整示例：自定义所有参数
python run_pipeline.py \
    --seed_samples 100 \
    --target_samples 1000 \
    --num_responses 3 \
    --model_name qwen-turbo
```

参数说明：
- `--seed_samples`: 从种子数据中读取的样本数量（默认: 50，设置为 0 表示读取全部）
- `--target_samples`: 目标生成的数据量（默认: 500）
- `--num_responses`: 每个样本生成的变体数（默认: 2）
- `--model_name`: 使用的模型名称（默认: qwen-plus，可选: qwen-turbo, qwen-max）

## 方法二：使用环境变量

```bash
# 设置环境变量
export SEED_SAMPLES=100

# 然后运行脚本
python run_pipeline.py
```

## 方法三：直接修改代码

编辑 `run_pipeline.py` 文件，找到以下行并修改：

```python
SEED_SAMPLES = 50  # 改为你想要的数字，或 None 表示读取全部
```

## 方法四：直接运行指令合成脚本

如果你想手动控制，可以直接运行 `instruction_synthesis.py`：

```bash
# 读取前 100 个样本
PYTHONPATH=. python src/instruction_synthesis.py \
    --input_path data/seed/seed_samples.json \
    --output_dir data/round1_synthesis \
    --max_samples 100 \
    --model_name qwen-plus \
    --num_threads 4 \
    --num_responses 2

# 读取全部样本（不设置 --max_samples 或设置为 0）
PYTHONPATH=. python src/instruction_synthesis.py \
    --input_path data/seed/seed_samples.json \
    --output_dir data/round1_synthesis \
    --max_samples 0 \
    --model_name qwen-plus
```

## 示例场景

### 场景1：快速测试（只处理10个样本）

```bash
python run_pipeline.py --seed_samples 10 --target_samples 50
```

### 场景2：中等规模（处理100个样本，生成500条数据）

```bash
python run_pipeline.py --seed_samples 100 --target_samples 500
```

### 场景3：大规模（处理全部种子样本，生成2000条数据）

```bash
python run_pipeline.py --seed_samples 0 --target_samples 2000 --num_responses 3
```

## 注意事项

1. **种子样本数量与最终数据量的关系**：
   - 第1轮：种子数 × 每样本变体数 = 第1轮生成数
   - 第2轮：第1轮高质量样本 × 每样本变体数 = 第2轮生成数
   - 最终数据会经过质量过滤，实际数量可能少于预期

2. **建议配置**：
   - 快速测试：`--seed_samples 10`
   - 标准运行：`--seed_samples 50`（默认）
   - 大规模生成：`--seed_samples 100` 或 `0`（全部）

3. **性能考虑**：
   - 种子样本越多，处理时间越长
   - 建议根据你的 API 配额和时间预算来设置

