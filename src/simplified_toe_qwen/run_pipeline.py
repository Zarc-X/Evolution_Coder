#!/usr/bin/env python3
"""
简化版 Tree-of-Evolution 完整流程运行脚本
目标：生成500条高质量数据
"""

import os
import sys
import subprocess
import logging
import argparse

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


def run_command(cmd, description):
    """运行命令并处理错误。"""
    logger.info(f"开始: {description}")
    logger.info(f"执行命令: {' '.join(cmd)}")
    
    # 设置 PYTHONPATH 以便导入模块
    env = os.environ.copy()
    env['PYTHONPATH'] = os.path.dirname(__file__)
    
    result = subprocess.run(cmd, cwd=os.path.dirname(__file__), env=env)
    
    if result.returncode != 0:
        logger.error(f"失败: {description}")
        sys.exit(1)
    
    logger.info(f"完成: {description}")
    return result


def parse_arguments():
    """解析命令行参数"""
    parser = argparse.ArgumentParser(
        description="简化版 Tree-of-Evolution 完整流程运行脚本"
    )
    parser.add_argument(
        "--seed_samples",
        type=int,
        default=None,
        help="从种子数据中读取的样本数量（默认: 50，设置为0表示读取全部）"
    )
    parser.add_argument(
        "--target_samples",
        type=int,
        default=500,
        help="目标生成的数据量（默认: 500）"
    )
    parser.add_argument(
        "--num_responses",
        type=int,
        default=2,
        help="每个样本生成的变体数（默认: 2）"
    )
    parser.add_argument(
        "--model_name",
        type=str,
        default="qwen-plus",
        help="使用的模型名称（默认: qwen-plus，可选: qwen-turbo, qwen-max）"
    )
    return parser.parse_args()


def main():
    """主流程：生成500条数据"""
    
    args = parse_arguments()
    
    # 配置参数
    TARGET_SAMPLES = args.target_samples
    # 从种子数据中读取的样本数量
    # 优先级：命令行参数 > 环境变量 > 默认值
    if args.seed_samples is not None:
        SEED_SAMPLES = args.seed_samples if args.seed_samples > 0 else None
    else:
        SEED_SAMPLES = os.getenv("SEED_SAMPLES")
        if SEED_SAMPLES:
            SEED_SAMPLES = int(SEED_SAMPLES) if int(SEED_SAMPLES) > 0 else None
        else:
            SEED_SAMPLES = 50  # 默认50个种子样本
    
    NUM_RESPONSES = args.num_responses
    MODEL_NAME = args.model_name
    
    logger.info(f"配置: 种子样本数={SEED_SAMPLES if SEED_SAMPLES else '全部'}, 目标数据量={TARGET_SAMPLES}, 每样本变体数={NUM_RESPONSES}, 模型={MODEL_NAME}")
    
    # 检查环境变量
    if not os.getenv("DASHSCOPE_API_KEY"):
        logger.error("请先设置环境变量 DASHSCOPE_API_KEY")
        logger.info("可以创建 .env 文件或运行: export DASHSCOPE_API_KEY=your_key")
        sys.exit(1)
    
    # 创建数据目录
    os.makedirs("data/seed", exist_ok=True)
    os.makedirs("data/round1_synthesis", exist_ok=True)
    os.makedirs("data/round1_complexity", exist_ok=True)
    os.makedirs("data/round1_diversity", exist_ok=True)
    os.makedirs("data/round2_synthesis", exist_ok=True)
    os.makedirs("data/round2_complexity", exist_ok=True)
    os.makedirs("data/round2_diversity", exist_ok=True)
    os.makedirs("data/final", exist_ok=True)
    
    # 检查种子数据
    seed_file = "data/seed/seed_samples.json"
    if not os.path.exists(seed_file):
        logger.warning(f"种子数据文件不存在: {seed_file}")
        logger.info("请创建种子数据文件，格式参考 README.md")
        logger.info("或者使用原始项目的种子数据")
        sys.exit(1)
    
    logger.info("=" * 60)
    logger.info("开始简化版 Tree-of-Evolution 流程")
    logger.info(f"目标: 生成约 {TARGET_SAMPLES} 条高质量数据")
    logger.info("=" * 60)
    
    # ========== 第1轮：初始演化 ==========
    logger.info("\n" + "=" * 60)
    logger.info("第1轮：初始演化（从种子数据）")
    logger.info("=" * 60)
    
    # Step 1: 指令合成
    cmd = [
        sys.executable, "src/instruction_synthesis.py",
        "--input_path", seed_file,
        "--output_dir", "data/round1_synthesis",
        "--model_name", MODEL_NAME,
        "--num_threads", "4",
        "--num_responses", str(NUM_RESPONSES),
        "--temperature", "1.0",
        "--max_tokens", "2048"
    ]
    # 只有当 SEED_SAMPLES 不为 None 时才添加 --max_samples 参数
    if SEED_SAMPLES is not None:
        cmd.extend(["--max_samples", str(SEED_SAMPLES)])
    run_command(cmd, "第1轮：指令合成")
    
    # 收集结果
    synthesis_file = "data/round1_synthesis/all_questions.json"
    if not os.path.exists(synthesis_file):
        # 查找实际生成的文件
        import glob
        files = glob.glob("data/round1_synthesis/all_questions_*.json")
        if files:
            synthesis_file = files[0]
        else:
            logger.error("未找到合成结果文件")
            sys.exit(1)
    
    # Step 2: 复杂度评分
    run_command([
        sys.executable, "src/complexity_scoring.py",
        "--input_path", synthesis_file,
        "--output_dir", "data/round1_complexity",
        "--model_name", MODEL_NAME,
        "--num_threads", "4",
        "--temperature", "0.0",
        "--max_tokens", "2048"
    ], "第1轮：复杂度评分")
    
    # 收集复杂度评分结果
    complexity_file = "data/round1_complexity/all_questions_w_complexity_scores.json"
    if not os.path.exists(complexity_file):
        import glob
        files = glob.glob("data/round1_complexity/all_questions_*_w_complexity_scores.json")
        if files:
            complexity_file = files[0]
        else:
            logger.error("未找到复杂度评分结果文件")
            sys.exit(1)
    
    # Step 3: 多样性评分
    run_command([
        sys.executable, "src/diversity_scoring.py",
        "--input_path", complexity_file,
        "--output_path", "data/round1_diversity/questions_w_diversity_scores.json",
        "--model_name", "Alibaba-NLP/gte-large-en-v1.5",
        "--batch_size", "10",
        "--device", "auto"
    ], "第1轮：多样性评分")
    
    diversity_file = "data/round1_diversity/questions_w_diversity_scores.json"
    
    # ========== 第2轮：优化驱动演化 ==========
    logger.info("\n" + "=" * 60)
    logger.info("第2轮：优化驱动演化")
    logger.info("=" * 60)
    
    # 计算需要处理的样本数（以达到目标数量）
    import json
    with open(diversity_file, 'r', encoding='utf-8') as f:
        round1_data = json.load(f)
    
    # 过滤高质量样本（复杂度 >= 6 且多样性 >= 0.5）
    high_quality = [
        item for item in round1_data
        if item.get("self complexity score", 0) >= 6.0
        and item.get("self diversity score", 0) >= 0.5
    ]
    
    logger.info(f"第1轮生成了 {len(round1_data)} 个样本")
    logger.info(f"高质量样本（复杂度>=6, 多样性>=0.5）: {len(high_quality)} 个")
    
    # 保存高质量样本用于第2轮
    high_quality_file = "data/round1_diversity/high_quality_samples.json"
    with open(high_quality_file, 'w', encoding='utf-8') as f:
        json.dump(high_quality, f, ensure_ascii=False, indent=2)
    
    # 限制样本数以控制最终数量
    max_round2_samples = min(len(high_quality), TARGET_SAMPLES // (NUM_RESPONSES * 2))
    if max_round2_samples < len(high_quality):
        high_quality = high_quality[:max_round2_samples]
        with open(high_quality_file, 'w', encoding='utf-8') as f:
            json.dump(high_quality, f, ensure_ascii=False, indent=2)
        logger.info(f"限制第2轮处理样本数为: {max_round2_samples}")
    
    # Step 4: 优化驱动指令合成
    run_command([
        sys.executable, "src/instruction_synthesis.py",
        "--input_path", high_quality_file,
        "--output_dir", "data/round2_synthesis",
        "--model_name", MODEL_NAME,
        "--num_threads", "4",
        "--num_responses", str(NUM_RESPONSES),
        "--temperature", "1.0",
        "--max_tokens", "2048",
        "--opt_evo"
    ], "第2轮：优化驱动指令合成")
    
    # 收集第2轮结果
    round2_synthesis_file = "data/round2_synthesis/all_questions.json"
    if not os.path.exists(round2_synthesis_file):
        import glob
        files = glob.glob("data/round2_synthesis/all_questions_*.json")
        if files:
            round2_synthesis_file = files[0]
        else:
            logger.error("未找到第2轮合成结果文件")
            sys.exit(1)
    
    # Step 5: 第2轮复杂度评分
    run_command([
        sys.executable, "src/complexity_scoring.py",
        "--input_path", round2_synthesis_file,
        "--output_dir", "data/round2_complexity",
        "--model_name", MODEL_NAME,
        "--num_threads", "4",
        "--temperature", "0.0",
        "--max_tokens", "2048"
    ], "第2轮：复杂度评分")
    
    # 收集第2轮复杂度结果
    round2_complexity_file = "data/round2_complexity/all_questions_w_complexity_scores.json"
    if not os.path.exists(round2_complexity_file):
        import glob
        files = glob.glob("data/round2_complexity/all_questions_*_w_complexity_scores.json")
        if files:
            round2_complexity_file = files[0]
        else:
            logger.error("未找到第2轮复杂度评分结果文件")
            sys.exit(1)
    
    # Step 6: 第2轮多样性评分
    run_command([
        sys.executable, "src/diversity_scoring.py",
        "--input_path", round2_complexity_file,
        "--output_path", "data/round2_diversity/questions_w_diversity_scores.json",
        "--model_name", "Alibaba-NLP/gte-large-en-v1.5",
        "--batch_size", "10",
        "--device", "auto"
    ], "第2轮：多样性评分")
    
    # ========== 最终结果整理 ==========
    logger.info("\n" + "=" * 60)
    logger.info("整理最终结果")
    logger.info("=" * 60)
    
    final_file = "data/round2_diversity/questions_w_diversity_scores.json"
    with open(final_file, 'r', encoding='utf-8') as f:
        final_data = json.load(f)
    
    # 再次过滤高质量样本
    final_high_quality = [
        item for item in final_data
        if item.get("self complexity score", 0) >= 6.5
        and item.get("self diversity score", 0) >= 0.6
    ]
    
    # 限制到目标数量
    if len(final_high_quality) > TARGET_SAMPLES:
        # 按复杂度+多样性总分排序，取前N个
        final_high_quality.sort(
            key=lambda x: x.get("self complexity score", 0) + x.get("self diversity score", 0) * 10,
            reverse=True
        )
        final_high_quality = final_high_quality[:TARGET_SAMPLES]
    
    # 保存最终结果
    output_file = f"data/final/final_dataset_{len(final_high_quality)}.json"
    with open(output_file, 'w', encoding='utf-8') as f:
        json.dump(final_high_quality, f, ensure_ascii=False, indent=2)
    
    logger.info("=" * 60)
    logger.info("流程完成！")
    logger.info(f"最终生成了 {len(final_high_quality)} 条高质量数据")
    logger.info(f"结果保存在: {output_file}")
    logger.info("=" * 60)
    
    # 打印统计信息
    if final_high_quality:
        complexities = [item.get("self complexity score", 0) for item in final_high_quality]
        diversities = [item.get("self diversity score", 0) for item in final_high_quality]
        
        logger.info("\n统计信息:")
        logger.info(f"  平均复杂度: {sum(complexities)/len(complexities):.2f}")
        logger.info(f"  平均多样性: {sum(diversities)/len(diversities):.2f}")
        logger.info(f"  复杂度范围: {min(complexities):.2f} - {max(complexities):.2f}")
        logger.info(f"  多样性范围: {min(diversities):.2f} - {max(diversities):.2f}")


if __name__ == "__main__":
    main()

