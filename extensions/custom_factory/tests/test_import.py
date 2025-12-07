def test_import_custom_factory():
    from extensions.custom_factory import custom_basegp_residual_factory

    assert custom_basegp_residual_factory is not None
