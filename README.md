# Evolution-Coder 完整系统

> 完整的Qwen2.5-Coder模型微调与评估系统

##  快速导航

- [项目特点](#项目特点)
- [项目结构](#项目结构)  
- [快速开始](#快速开始)
- [使用说明](#使用说明)
- [模块化优点](#模块化优点)
- [详细文档](#详细文档)

##  项目特点

- ✅ **模块化设计** - 代码从单个2000+行文件拆分为6个独立模块
- ✅ **易于维护** - 每个模块专注一个功能，便于修改和扩展
- ✅ **高度可测试** - 可单独测试每个模块
- ✅ **配置集中管理** - 所有配置在 `src/config/settings.py` 中
- ✅ **完整功能** - 模型加载、微调、评估、问答一体化
- ✅ **Web界面** - 使用Gradio提供友好的Web UI

##  项目结构

```
Evolution_Coder/
├── src/                              # 源代码目录
│   ├── config/                      # 配置模块 (集中管理)
│   │   ├── __init__.py
│   │   └── settings.py              # API_CONFIG, DEFAULT_CONFIG
│   │
│   ├── utils/                       # 工具函数模块 (复用代码)
│   │   ├── logger.py                # 日志系统
│   │   ├── api_helper.py            # API调用
│   │   ├── code_tools.py            # 代码处理
│   │   └── qa_interface.py          # 问答接口
│   │
│   ├── models/                      # 模型加载模块
│   │   └── loader.py                # 模型加载逻辑
│   │
│   ├── training/                    # 训练模块
│   │   └── trainer.py               # 训练线程和逻辑
│   │
│   ├── evaluation/                  # 评估模块
│   │   └── evaluator.py             # 评估线程和逻辑
│   │
│   └── ui/                          # UI界面模块
│       └── interface.py             # Gradio界面
│
├── main.py                          # 主程序入口
├── PROJECT_STRUCTURE.md             # 详细文档
├── datasets/
├── models/
└── requirements.txt
```

##  快速开始

### 1. 环境要求

```bash
Python 3.8+
CUDA 11.0+ (可选，用于GPU加速)
```

### 2. 安装依赖

```bash
# 安装依赖
pip install -r requirements.txt
```

### 3. 运行程序

```bash
python main.py
```

然后访问 `http://localhost:7860`

##  模块说明

### 1. `src/config/` - 配置模块
**职责**: 集中管理所有配置和常量

```python
# settings.py
API_CONFIG = {
    "qwen_32b_api_url": "...",
    "api_key": "..."
}

DEFAULT_CONFIG = {
    "model_path": "./models/Qwen2.5-Coder-0.5B-Instruct",
    "num_epochs": 3,
    ...
}
```

**优点**: 
- 修改配置无需改动其他文件
- 所有参数一目了然
- 易于版本管理

### 2. `src/utils/` - 工具模块
**职责**: 提供可复用的工具函数

- **logger.py** - 日志系统 (`log()`, `LogCollector`)
- **api_helper.py** - API调用 (`call_qwen_api()`, `validate_code_with_14b()`)
- **code_tools.py** - 代码处理 (`check_code_syntax()`, `extract_function_name()`)
- **qa_interface.py** - 问答接口 (`generate_code_with_local_model()`)

**优点**:
- 函数独立，可在其他项目中复用
- 测试隔离，易于编写单元测试
- 代码变更只需修改一处

### 3. `src/models/` - 模型加载模块
**职责**: 管理模型加载和状态

- `load_model_interface()` - 加载模型
- `get_model()` - 获取模型实例
- `is_model_loaded()` - 检查加载状态

**优点**:
- 全局模型实例管理
- 易于扩展多模型支持

### 4. `src/training/` - 训练模块
**职责**: 处理模型训练逻辑

- `TrainingThread` - 异步训练线程
- `generate_mbpp_training_data()` - 生成训练数据
- `start_training_interface()` - 训练界面

**优点**:
- 异步执行，不阻断UI
- 可独立测试训练流程

### 5. `src/evaluation/` - 评估模块
**职责**: 模型评估和结果对比

- `EvaluationThread` - 异步评估线程
- `get_comparison_results()` - 获取对比结果

**优点**:
- 隔离评估逻辑
- 易于添加新的评估指标

### 6. `src/ui/` - UI界面模块
**职责**: Gradio界面定义

- `create_interface()` - 创建完整UI
- `update_system_info()` - 更新系统信息

**优点**:
- UI和业务逻辑分离
- 易于更换UI框架

##  使用示例

### 生成代码
```
输入: "Write a function to add two numbers"
输出: 完整的Python函数代码
```

### 自我演化（自动微调）
```
输入: "自我演化"
执行流程:
  1. 读取MBPP数据集
  2. 调用API生成训练数据对
  3. 验证代码质量
  4. 微调模型
  5. 保存新模型
```

### 模型评估
```
对比原始模型和微调后模型的性能
显示通过率提升百分比
```

##  模块化优点详解

### 1. 可维护性 
- 代码行数从2000+降至200-300行/文件
- 修改一个模块不影响其他模块
- 易于查找和定位问题

### 2. 可测试性 
```python
# 可单独测试logger模块
def test_logger():
    from src.utils import log, log_collector
    log("测试")
    assert len(log_collector.logs) > 0

# 可单独测试API模块
def test_api():
    from src.utils import call_qwen_api
    success, code = call_qwen_api(...)
    assert success == True
```

### 3. 可扩展性 
```python
# 添加新的模型只需修改 loader.py
# 添加新的评估指标只需修改 evaluator.py
# 添加新的UI只需修改 interface.py
```

### 4. 代码复用 
```python
# utils 中的工具函数可在其他项目中使用
# 不需要复制整个 code.py 文件
```

### 5. 团队协作 
```
可以并行开发:
- A负责训练模块
- B负责评估模块
- C负责UI界面
- 基于接口协议，互不干扰
```

##  配置修改指南

### 修改模型路径
```python
# src/config/settings.py
DEFAULT_CONFIG = {
    "model_path": "./my_models/my_qwen",  # 修改这里
    ...
}
```

### 修改训练参数
```python
# src/config/settings.py
DEFAULT_CONFIG = {
    "learning_rate": 3e-4,  # 修改学习率
    "batch_size": 8,        # 修改批大小
    "num_epochs": 5,        # 修改轮数
}
```

### 修改API密钥
```python
# src/config/settings.py
API_CONFIG = {
    "api_key": "your_new_key_here",  # 修改这里
}
```

##  性能对比

| 指标 | 原始版本 | 模块化版本 |
|------|--------|---------|
| 总代码行数 | 3000+ | 约4500 (分散) |
| 最大文件行数 | 2000+ | 300-400 |
| 重用代码比例 | 0% | 60% |
| 测试覆盖难度 | 非常难 | 相对容易 |
| 功能复用难度 | 非常难 | 相对容易 |
| 新增功能工作量 | 高 | 低 |
|耦合性 | 极高 | 低 |

##  故障排除

### 缺少依赖库
```bash
pip install -r requirements.txt
```

### GPU未检测到
程序会自动使用CPU运行，但速度会较慢

### 模型加载失败
- 检查模型路径是否正确
- 确保模型文件完整
- 查看日志输出获取详细错误信息

### 模型测试准确率异常
- 检查测试数据集datasets\human-eval-v2-20210705.jsonl是否正确
- 从main.py退出，运行位于根目录的code.py

##  后续改进方向

- [ ] 添加单元测试框架
- [ ] 支持配置文件（JSON/YAML）
- [ ] 添加更多日志级别
- [ ] 支持多个模型同时加载
- [ ] 添加模型版本管理
- [ ] 优化内存使用
- [ ] 添加数据持久化层
- [ ] 支持分布式训练
- [ ] 支持更多数据集的测试（LiveCodeBench-main、evalplus-master、bigcodebench-main）
- [ ] 将ToE指令生成模块加入到前端界面
