 
import pytest
import yaml
 
from marvel_characters.config import ProjectConfig, Tags
 
 
@pytest.fixture
def valid_config_dict():
    return {
        "num_features": ["Height", "Weight"],
        "cat_features": ["Universe", "Gender"],
        "target": "Alive",
        "parameters": {"n_estimators": 100},
        "experiment_name_basic": "/exp/basic",
        "experiment_name_custom": "/exp/custom",
        "dev": {"catalog_name": "dev_catalog", "schema_name": "dev_schema"},
        "acc": {"catalog_name": "acc_catalog", "schema_name": "acc_schema"},
        "prd": {"catalog_name": "prd_catalog", "schema_name": "prd_schema"},
    }
 
 
def test_from_yaml_loads_dev_environment_by_default(tmp_path, valid_config_dict):
    config_file = tmp_path / "config.yml"
    config_file.write_text(yaml.dump(valid_config_dict))
 
    config = ProjectConfig.from_yaml(str(config_file))  # default env="dev"
 
    assert config.catalog_name == "dev_catalog"
    assert config.schema_name == "dev_schema"
    assert config.target == "Alive"
 
 
def test_from_yaml_loads_requested_environment(tmp_path, valid_config_dict):
    config_file = tmp_path / "config.yml"
    config_file.write_text(yaml.dump(valid_config_dict))
 
    config = ProjectConfig.from_yaml(str(config_file), env="prd")
 
    assert config.catalog_name == "prd_catalog"
    assert config.schema_name == "prd_schema"
 
 
def test_from_yaml_invalid_env_raises_value_error(tmp_path, valid_config_dict):
    config_file = tmp_path / "config.yml"
    config_file.write_text(yaml.dump(valid_config_dict))
 
    with pytest.raises(ValueError, match="Invalid environment"):
        ProjectConfig.from_yaml(str(config_file), env="staging")
 
 
def test_project_config_missing_required_field_raises(valid_config_dict):
    incomplete = {k: v for k, v in valid_config_dict.items() if k != "target"}
    incomplete["catalog_name"] = "dev_catalog"
    incomplete["schema_name"] = "dev_schema"
    del incomplete["dev"], incomplete["acc"], incomplete["prd"]
 
    with pytest.raises(Exception):  # pydantic.ValidationError
        ProjectConfig(**incomplete)
 
 
def test_tags_to_dict_includes_run_id_when_present():
    tags = Tags(git_sha="abc123", branch="main", run_id="run-1")
    assert tags.to_dict() == {"git_sha": "abc123", "branch": "main", "run_id": "run-1"}
 
 
def test_tags_to_dict_excludes_run_id_when_none():
    tags = Tags(git_sha="abc123", branch="main")
    result = tags.to_dict()
    assert "run_id" not in result
    assert result == {"git_sha": "abc123", "branch": "main"}
