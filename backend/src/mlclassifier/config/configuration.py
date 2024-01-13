from mlclassifier.constants import *
from mlclassifier.utils.common import read_yaml, create_directories, save_json
from mlclassifier.entity import DataIngestionConfig, DataLoaderConfig
import os
from pathlib import Path


class ConfigurationManager:
    def __init__(
            self,
            config_filepath=CONFIG_FILE_PATH,
            params_filepath=PARAMS_FILE_PATH):

        self.config = read_yaml(config_filepath)
        self.params = read_yaml(params_filepath)

        create_directories([self.config.artifacts_root])

    def get_data_ingestion_config(self) -> DataIngestionConfig:
        config = self.config.data_ingestion

        create_directories([config.root_dir])

        data_ingestion_config = DataIngestionConfig(
            root_dir=config.root_dir,
            source_URL=config.source_URL,
            local_data_file=config.local_data_file,
            unzip_dir=config.unzip_dir
        )

        return data_ingestion_config

    def get_data_loader_config(self) -> DataLoaderConfig:
        config = self.config.data_loader

        data_ingestion_config = DataLoaderConfig(
            root_dir=config.root_dir,
        )

        return data_ingestion_config
