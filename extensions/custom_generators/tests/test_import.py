def test_import_custom_generators():
    # import the module file directly to avoid package-level __init__ imports
    import importlib.util
    from pathlib import Path

    module_path = Path(__file__).resolve().parents[1] / "custom_pool_based_generator.py"
    spec = importlib.util.spec_from_file_location(
        "custom_pool_based_generator", str(module_path)
    )
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    assert module is not None
