def test_import_custom_mean():
    # basic import smoke test
    from extensions.custom_mean import custom_basegp_prior_mean

    assert custom_basegp_prior_mean is not None
