from __future__ import annotations

import json
import logging
import os
import traceback
from datetime import datetime
from typing import Any, Dict, Optional, Tuple


_ROOT_CONFIGURED = False

def setup_experiment_logger(
    component: str,
    run_name: Optional[str] = None,
    log_dir: str = "logs",
    config: Optional[Dict[str, Any]] = None,
) -> Tuple[logging.Logger, str, str]:
    final_run_name = run_name or datetime.now().strftime("exp_%Y%m%d_%H%M%S")
    os.makedirs(log_dir, exist_ok=True)
    log_path = os.path.join(log_dir, f"{final_run_name}.log")

    global _ROOT_CONFIGURED
    if not _ROOT_CONFIGURED:
        logging.basicConfig(
            level=logging.INFO,
            format="%(asctime)s | %(levelname)s | %(name)s | %(message)s",
        )
        logging.captureWarnings(True)
        _ROOT_CONFIGURED = True

    logger_name = f"tf_mil.{component}.{final_run_name}"
    logger = logging.getLogger(logger_name)
    logger.setLevel(logging.INFO)
    # propagate=True (default): messages reach root → stderr.
    # FileHandler keeps this stage's messages in its own log file.

    if not logger.handlers:
        file_handler = logging.FileHandler(log_path, mode="a", encoding="utf-8")
        root_fmt = logging.getLogger().handlers[0].formatter
        file_handler.setFormatter(root_fmt)
        logger.addHandler(file_handler)

    logger.info("run_start component=%s run_name=%s", component, final_run_name)
    if config is not None:
        logger.info("config_json=%s", json.dumps(config, ensure_ascii=False, sort_keys=True))
    return logger, log_path, final_run_name


def log_exception(logger: logging.Logger, exc: Exception) -> None:
    logger.error("exception=%s", str(exc))
    logger.error("traceback=%s", traceback.format_exc())
