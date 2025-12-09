#!/usr/bin/env python3
"""
直接测试数据库API是否被调用
"""

import sys
import os
sys.path.insert(0, 'extensions/custom_generators')

from custom_pool_based_generator import CustomPoolBasedGenerator
from aepsych.server import AEPsychServer
from aepsych.config import Config
import torch
import tempfile

def test_direct_db_access():
    """直接测试数据库API是否能被调用"""
    
    print("🔬 直接数据库API访问测试")
    print("="*50)
    
    # 使用现有的数据库
    db_path = "databases/default.db"
    
    # 创建服务器实例
    server = AEPsychServer(database_path=db_path)
    
    # 创建generator实例
    pool_points = torch.tensor([[0, 0], [0, 1], [1, 0], [1, 1], [2, 0], [2, 1]], dtype=torch.float32)
    lb = torch.tensor([0, 0], dtype=torch.float32)
    ub = torch.tensor([2, 1], dtype=torch.float32)
    
    from botorch.acquisition import qUpperConfidenceBound
    acqf = qUpperConfidenceBound
    
    generator = CustomPoolBasedGenerator(
        lb=lb,
        ub=ub,
        pool_points=pool_points,
        acqf=acqf,
        dim=2,
        allow_resampling=False,
        shuffle=True
    )
    
    print("✅ Generator创建成功")
    
    # 检查是否有设置服务器的方法
    if hasattr(generator, 'set_aepsych_server'):
        generator.set_aepsych_server(server)
        print("✅ 成功设置服务器实例")
    else:
        print("❌ Generator没有set_aepsych_server方法")
        print("可用方法:", [m for m in dir(generator) if not m.startswith('_')])
        return
    
    # 插入一些测试数据到数据库
    print(f"\n📊 使用现有数据库: {db_path}")
    
    # 检查数据库中现有的数据
    try:
        result = server.db.execute_sql_query("SELECT COUNT(*) as count FROM param_data", ())
        if result:
            count = result[0][0] if result[0] else 0
            print(f"数据库中现有参数记录数: {count}")
    except Exception as e:
        print(f"查询数据库错误: {e}")
    
    # 直接测试数据库API调用
    print("\n🔍 测试数据库API调用...")
    
    try:
        history = generator._get_sampling_history_from_server()
        if history is not None and len(history) > 0:
            print(f"✅ 数据库API成功调用！获取到 {len(history)} 个历史点")
            print(f"历史点: {history}")
        else:
            print("⚠️  数据库API调用成功但没有获取到数据")
    except Exception as e:
        print(f"❌ 数据库API调用失败: {e}")
    
    # 不清理现有数据库
    
    print("\n✅ 测试完成")

if __name__ == "__main__":
    test_direct_db_access()