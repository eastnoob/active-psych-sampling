#!/usr/bin/env python
"""快速测试analyze_phase1模块"""

import sys
from pathlib import Path

# 添加路径
sys.path.insert(0, str(Path(__file__).parent))

try:
    from analyze_phase1 import Phase1DataAnalyzer

    print("✅ 成功导入Phase1DataAnalyzer")

    # 检查是否有样本数据
    sample_data = (
        Path(__file__).parent
        / "sample"
        / "202511182225"
        / "result"
        / "combined_results.csv"
    )
    if sample_data.exists():
        print(f"✅ 找到样本数据: {sample_data}")

        # 尝试分析
        analyzer = Phase1DataAnalyzer(
            data_csv_path=str(sample_data), subject_col="subject", response_col="y"
        )
        print("✅ 成功创建分析器")

        # 运行分析
        analyzer.analyze()
        print("✅ 分析完成")

        # 生成Phase 2配置
        phase2_config = analyzer.generate_phase2_config(
            n_subjects=20, trials_per_subject=25, lambda_adjustment=1.2
        )
        print("✅ Phase 2配置生成完成")

        # 导出报告
        output_dir = Path(__file__).parent / "test_output"
        output_dir.mkdir(exist_ok=True)

        analyzer.export_report(
            phase2_config=phase2_config,
            output_dir=str(output_dir),
            prefix="phase1",
            report_format="md",
        )
        print(f"✅ 报告已导出到 {output_dir}")

        # 列出生成的文件
        print("\n生成的文件：")
        for f in sorted(output_dir.glob("*")):
            size = f.stat().st_size if f.is_file() else "DIR"
            print(
                f"  - {f.name} ({size} bytes)"
                if isinstance(size, int)
                else f"  - {f.name} ({size})"
            )

    else:
        print(f"❌ 未找到样本数据: {sample_data}")
        sys.exit(1)

except Exception as e:
    print(f"❌ 错误: {e}")
    import traceback

    traceback.print_exc()
    sys.exit(1)
