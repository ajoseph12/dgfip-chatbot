"""Smoke tests — keep `make test` green from day one."""

from pathlib import Path


def test_package_imports():
    import dgfip_chatbot

    assert dgfip_chatbot.__version__


def test_config_resolves_data_paths():
    from dgfip_chatbot.config import settings

    assert settings.kb_path.name == "info_particulier_impot.csv"
    assert settings.questions_path.name == "questions_fiches_fip.csv"
    assert isinstance(settings.raw_data_dir, Path)


def test_provided_csvs_present():
    from dgfip_chatbot.config import settings

    assert settings.kb_path.exists(), "expected the KB CSV under data/raw/"
    assert settings.questions_path.exists(), "expected the questions CSV under data/raw/"
