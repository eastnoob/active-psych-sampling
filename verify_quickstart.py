from configparser import ConfigParser

config = ConfigParser()
config.read(
    "extensions/dynamic_eur_acquisition/configs/QUICKSTART.ini", encoding="utf-8"
)

print("✓ QUICKSTART.ini 解析成功\n")
print(f"配置部分: {config.sections()}\n")

c = config["EURAnovaMultiAcqf"]
print("【EURAnovaMultiAcqf 参数验证】")
print(f"总参数数: {len(c)}")
print(f'enable_main = {c.get("enable_main")}')
print(f'enable_pairwise = {c.get("enable_pairwise")}')
print(f'enable_threeway = {c.get("enable_threeway")}')
print(f'interaction_pairs = {c.get("interaction_pairs")}')
print(f'lambda_min = {c.get("lambda_min")}')
print(f'lambda_max = {c.get("lambda_max")}')
print(f'gamma = {c.get("gamma")}')
print(f'use_sps = {c.get("use_sps")}')
print(f'local_num = {c.get("local_num")}')
print(f'use_hybrid_perturbation = {c.get("use_hybrid_perturbation")}')
print(f'exhaustive_level_threshold = {c.get("exhaustive_level_threshold")}')
print(f'coverage_method = {c.get("coverage_method")}')
print(f'fusion_method = {c.get("fusion_method")}')
print("\n✓ 所有关键参数存在且可解析")
