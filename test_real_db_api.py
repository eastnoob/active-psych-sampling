#!/usr/bin/env python3
"""
真实AEPsych服务器数据库API测试
验证CustomPoolBasedGenerator的数据库集成功能
"""

import sys
import os
import tempfile
import time
from pathlib import Path

# 添加路径
sys.path.insert(0, 'extensions/custom_generators')
sys.path.insert(0, 'extensions/dynamic_eur_acquisition')

def test_real_aepsych_db():
    """测试真实的AEPsych服务器环境下的数据库API"""
    
    print("🔬 真实AEPsych服务器数据库API测试")
    print("="*50)
    
    # 配置内容
    config_content = """
[common]
parnames = [x1, x2]
stimuli_per_trial = 1
outcome_types = [continuous]
strategy_names = [init_strat, opt_strat]
lb = [0, 0]
ub = [2, 1]

[x1]
par_type = categorical
choices = [0, 1, 2]
lb = 0
ub = 2

[x2]
par_type = categorical
choices = [0, 1]
lb = 0
ub = 1

[init_strat]
min_asks = 2
generator = ManualGenerator

[ManualGenerator]
points = [[0, 0], [1, 1]]

[opt_strat]
min_asks = 4
max_asks = 6
generator = CustomPoolBasedGenerator
model = GPRegressionModel
acqf = qUpperConfidenceBound
refit_every = 1

[CustomPoolBasedGenerator]
pool_points = [[0, 0], [0, 1], [1, 0], [1, 1], [2, 0], [2, 1]]
acqf = qUpperConfidenceBound
allow_resampling = False
shuffle = True

[qUpperConfidenceBound]
beta = 2.0
"""
    
    try:
        # 导入必要的模块
        from aepsych.config import Config
        from aepsych.server import AEPsychServer
        from custom_pool_based_generator import CustomPoolBasedGenerator
        
        # 注册组件
        Config.register_object(CustomPoolBasedGenerator)
        
        # 创建配置
        config = Config(config_str=config_content)
        print("✅ 配置创建成功")
        
        # 创建临时数据库文件
        db_path = "test_db_api.db"
        socket_path = "test_db_api.sock"
        
        # 清理可能存在的文件
        for path in [db_path, socket_path]:
            if os.path.exists(path):
                os.remove(path)
        
        # 创建服务器
        server = AEPsychServer(
            socket=socket_path,
            database_path=db_path
        )
        print("✅ AEPsych服务器创建成功")
        print(f"📁 数据库文件: {db_path}")
        
        # 通过消息方式配置服务器
        setup_msg = {
            "type": "setup",
            "message": {
                "config_str": config_content
            }
        }
        
        response = server.handle_request(setup_msg)
        print("✅ 服务器配置完成")
        print(f"配置响应: {response}")
        
        # 获取generator实例并设置服务器引用
        try:
            if hasattr(server, 'strat') and server.strat is not None:
                # 从SequentialStrategy中获取当前策略
                current_strat = server.strat.strats[server.strat._strat_idx] if hasattr(server.strat, 'strats') else server.strat
                if hasattr(current_strat, 'generator'):
                    generator = current_strat.generator
                    if hasattr(generator, 'set_aepsych_server'):
                        generator.set_aepsych_server(server)
                        print("✅ 成功设置服务器实例到generator")
                    else:
                        print("⚠️  Generator没有set_aepsych_server方法")
                else:
                    print("⚠️  Strategy没有generator属性")
            else:
                print("⚠️  Server没有strategy")
        except Exception as e:
            print(f"⚠️  设置服务器实例失败: {e}")
        
        # 模拟实验过程
        print("\n🎯 开始模拟实验...")
        
        all_points = []
        
        # 执行完整的采样循环
        for i in range(6):  # 总共6次采样
            print(f"\n=== 采样迭代 {i+1} ===")
            
            # 询问下一个点
            ask_msg = {"type": "ask", "message": {}}
            result = server.handle_request(ask_msg)
            print(f"询问结果: {result}")
            
            if 'config' in result:
                config_point = result['config']
                print(f"选中点: {config_point}")
                all_points.append(config_point)
                
                # 检查重复
                is_duplicate = False
                current_point = tuple(sorted(config_point.items()))
                
                for j, prev_point in enumerate(all_points[:-1]):
                    if tuple(sorted(prev_point.items())) == current_point:
                        is_duplicate = True
                        print(f"❗ 发现重复！与点{j+1}相同")
                        break
                
                if not is_duplicate:
                    print("✅ 点唯一")
                
                # 模拟响应
                outcome = 1.0 + 0.2 * i  # 模拟响应
                tell_msg = {
                    "type": "tell",
                    "message": {
                        "config": config_point,
                        "outcome": outcome
                    }
                }
                tell_result = server.handle_request(tell_msg)
                print(f"告知响应 {outcome}: {tell_result}")
        
        print("\n📊 最终结果分析...")
        print(f"总采样点数: {len(all_points)}")
        
        # 手动检查重复
        unique_points = []
        duplicates = 0
        
        for i, point in enumerate(all_points):
            point_tuple = tuple(sorted(point.items()))
            if point_tuple in [tuple(sorted(p.items())) for p in unique_points]:
                duplicates += 1
                print(f"重复点{i+1}: {point}")
            else:
                unique_points.append(point)
        
        print(f"唯一点数: {len(unique_points)}")
        print(f"重复点数: {duplicates}")
        
        if duplicates == 0:
            print("🎉 SUCCESS: 数据库API修复完全有效！")
        else:
            print(f"⚠️ WARNING: 仍有 {duplicates} 个重复点")
        
        # 显示所有点
        print("\n所有采样点:")
        for i, point in enumerate(all_points):
            coord = [point.get(f'x{j}', 0) for j in [1, 2]]
            print(f"  点{i+1}: {coord}")
        
        # 数据库查询验证
        print("\n🔍 数据库内容验证:")
        if hasattr(server, 'db') and server.db is not None:
            try:
                query = "SELECT COUNT(*) FROM param_data"
                count_result = server.db.execute_sql_query(query, {})
                print(f"数据库中参数记录数: {count_result[0][0] if count_result else 0}")
                
                query2 = "SELECT COUNT(DISTINCT iteration_id) FROM param_data"  
                iter_result = server.db.execute_sql_query(query2, {})
                print(f"数据库中迭代数: {iter_result[0][0] if iter_result else 0}")
                
            except Exception as e:
                print(f"数据库查询错误: {e}")
        
        # 清理
        server = None
        for path in [db_path, socket_path]:
            if os.path.exists(path):
                try:
                    os.remove(path)
                except:
                    pass
        
        print("\n✅ 测试完成！")
        
    except Exception as e:
        print(f"❌ 测试失败: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    test_real_aepsych_db()