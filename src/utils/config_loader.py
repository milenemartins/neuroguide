"""
Carregador de configurações do projeto
"""
import yaml
from pathlib import Path
from typing import Dict, Any


class ConfigLoader:
    """Classe para carregar e acessar configurações do projeto"""

    def __init__(self, config_path: str = None):
        """
        Inicializa o carregador de configurações

        Args:
            config_path: Caminho para o arquivo de configuração YAML
        """
        if config_path is None:
            # Busca o config.yaml na raiz do projeto
            project_root = Path(__file__).parent.parent.parent
            config_path = project_root / "config" / "config.yaml"

        self.config_path = Path(config_path)
        self.config = self._load_config()

    def _load_config(self) -> Dict[str, Any]:
        """Carrega o arquivo de configuração YAML"""
        if not self.config_path.exists():
            raise FileNotFoundError(f"Arquivo de configuração não encontrado: {self.config_path}")

        with open(self.config_path, 'r', encoding='utf-8') as f:
            return yaml.safe_load(f)

    def get(self, key: str, default: Any = None) -> Any:
        """
        Obtém um valor de configuração usando notação de ponto

        Args:
            key: Chave da configuração (ex: 'model.input_shape')
            default: Valor padrão se a chave não existir

        Returns:
            Valor da configuração
        """
        keys = key.split('.')
        value = self.config

        for k in keys:
            if isinstance(value, dict) and k in value:
                value = value[k]
            else:
                return default

        return value

    def get_paths(self) -> Dict[str, Path]:
        """Retorna os caminhos principais do projeto"""
        project_root = Path(__file__).parent.parent.parent
        paths = self.config.get('paths', {})

        return {
            'project_root': project_root,
            'data_dir': project_root / paths.get('data_dir', 'data'),
            'models_dir': project_root / paths.get('models_dir', 'models'),
            'logs_dir': project_root / paths.get('logs_dir', 'logs'),
        }

    def __getitem__(self, key: str) -> Any:
        """Permite acesso usando config['key']"""
        return self.get(key)


# Instância global de configuração
config = ConfigLoader()
