import atexit
import logging
import os
import shutil
import tempfile
from dataclasses import dataclass
from pathlib import Path

from daidai.logs import get_logger
from daidai.types import (
    ArtifactCacheStrategy as CacheStrategy,
)

logger = get_logger(__name__)


@dataclass(frozen=True)
class DaiDaiConfig:
    cache_dir: Path
    cache_dir_tmp: Path
    force_download: bool
    log_level: str
    cache_strategy: CacheStrategy

    @classmethod
    def from_env(cls) -> "DaiDaiConfig":
        cache_strategy = cls._validate_cache_strategy_from_env(
            "DAIDAI_DEFAULT_CACHE_STRATEGY", CacheStrategy.ON_DISK
        )
        cache_dir = cls._validate_path_from_env(
            "DAIDAI_CACHE_DIR",
            Path.home() / ".daidai" / "cache",
            build=cache_strategy == CacheStrategy.ON_DISK,
        )
        cache_dir_tmp = cls._validate_path_from_env(
            "DAIDAI_CACHE_DIR_TMP",
            Path(tempfile.mkdtemp(prefix="daidai-")),
            build=cache_strategy == CacheStrategy.ON_DISK_TEMP,
        )
        if not cache_dir_tmp.is_dir():
            logger.error(f"Temporary cache is not a directory: {cache_dir_tmp}")
            raise ValueError(f"Temporary cache is not a directory: {cache_dir_tmp}")
        if any(cache_dir_tmp.iterdir()):
            logger.error(f"Temporary cache must be empty: {cache_dir_tmp}")
            raise ValueError(f"Temporary cache must be empty: {cache_dir_tmp}")

        force_download = cls._validate_force_download_from_env(
            "DAIDAI_FORCE_DOWNLOAD", False
        )

        def clean_up_tmp_dir():
            try:
                shutil.rmtree(cache_dir_tmp, ignore_errors=False)
                logger.info("Temporary cache directory cleaned up", path=cache_dir_tmp)
            except OSError:
                pass

        atexit.register(clean_up_tmp_dir)
        log_level = cls._validate_log_level_from_env("DAIDAI_LOG_LEVEL", "WARNING")
        cache_strategy = cls._validate_cache_strategy_from_env(
            "DAIDAI_DEFAULT_CACHE_STRATEGY", CacheStrategy.ON_DISK
        )
        logger.info(
            "Configuration initialized",
            cache_dir=cache_dir,
            cache_dir_tmp=cache_dir_tmp,
            log_level=log_level,
            cache_strategy=cache_strategy,
            force_download=force_download,
        )
        return cls(
            cache_dir=cache_dir,
            cache_dir_tmp=cache_dir_tmp,
            log_level=log_level,
            cache_strategy=cache_strategy,
            force_download=force_download,
        )

    @staticmethod
    def _validate_path_from_env(env_var: str, default: Path, build: bool) -> Path:
        path_str = os.getenv(env_var)
        path = Path(path_str) if path_str else default
        path = path.expanduser().resolve().absolute()
        if build and not path.exists():
            logger.info(f"Creating directory: {path}")
            try:
                path.mkdir(parents=True, exist_ok=True)
            except OSError as e:
                logger.error(
                    "Failed to create directory during config initialization",
                    env_var=env_var,
                    error=str(e),
                    path=str(path),
                )
                raise
        return path

    @staticmethod
    def _validate_log_level_from_env(env_var: str, default: str) -> str:
        level = os.getenv(env_var, default).upper()
        if level not in logging._nameToLevel:
            logger.error(f"Invalid log level: {level}")
            raise ValueError(f"Invalid log level: {level}")
        return level

    @staticmethod
    def _validate_cache_strategy_from_env(
        env_var: str, default: CacheStrategy
    ) -> CacheStrategy:
        """Parse and validate cache strategy."""
        cache_strategy = os.getenv(env_var)
        try:
            return CacheStrategy(cache_strategy.lower()) if cache_strategy else default
        except (ValueError, AttributeError) as e:
            logger.error(f"Invalid cache strategy: {cache_strategy}")
            raise ValueError(
                f"Invalid cache strategy: {cache_strategy}. "
                f"Valid options are: {', '.join(s.value for s in CacheStrategy)}"
            ) from e

    @staticmethod
    def _validate_force_download_from_env(env_var: str, default: bool) -> bool:
        """Parse en validate force download"""
        force_download = os.getenv(env_var)
        if not force_download:
            return default
        if force_download.lower() in ("true", "1"):
            return True
        if force_download.lower() in ("false", "0"):
            return False
        logger.error(f"Invalid value for force download: {force_download}")
        raise ValueError(f"Invalid value for force download: {force_download}")


CONFIG = DaiDaiConfig.from_env()
