"""
AEPsych Server管理模块
"""

from aepsych.server import AEPsychServer
from aepsych.config import Config
from pathlib import Path
from typing import Tuple, Optional
import logging


def initialize_server(config_path: Path) -> Tuple[AEPsychServer, Config]:
    """
    初始化AEPsych Server

    Args:
        config_path: 配置文件路径

    Returns:
        (server, config)元组
    """
    logging.info(f"Loading config from {config_path}")

    server = AEPsychServer()
    config = Config.from_file(str(config_path))
    server.configure(config)

    logging.info("Server initialized successfully")
    return server, config


def verify_server_components(server: AEPsychServer, config: Config) -> bool:
    """
    验证Server组件是否正确设置

    Args:
        server: AEPsych server实例
        config: Config实例

    Returns:
        验证是否通过
    """
    try:
        # 检查strat
        if not hasattr(server, '_strats') or len(server._strats) == 0:
            logging.error("Server has no strategies")
            return False

        strat = server._strats[0]

        # 检查model
        if not hasattr(strat, 'model'):
            logging.error("Strategy has no model")
            return False

        logging.info("Server components verified")
        return True

    except Exception as e:
        logging.error(f"Verification error: {e}")
        return False
