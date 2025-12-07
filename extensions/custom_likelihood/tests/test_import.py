def test_import_custom_likelihood():
    from extensions.custom_likelihood import custom_configurable_gaussian_likelihood

    assert custom_configurable_gaussian_likelihood is not None
