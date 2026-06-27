from unittest.mock import MagicMock, patch
 
from marvel_characters.utils import get_dbr_host, is_databricks
 
 
def test_is_databricks_true_when_env_var_set(monkeypatch):
    monkeypatch.setenv("DATABRICKS_RUNTIME_VERSION", "14.3")
    assert is_databricks() is True
 
 
def test_is_databricks_false_when_env_var_absent(monkeypatch):
    monkeypatch.delenv("DATABRICKS_RUNTIME_VERSION", raising=False)
    assert is_databricks() is False
 
 
@patch("marvel_characters.utils.WorkspaceClient")
def test_get_dbr_host_returns_host_from_workspace_client(mock_ws_client_cls):
    mock_instance = MagicMock()
    mock_instance.config.host = "https://my-workspace.databricks.com"
    mock_ws_client_cls.return_value = mock_instance
 
    result = get_dbr_host()
 
    assert result == "https://my-workspace.databricks.com"
    mock_ws_client_cls.assert_called_once()
 
