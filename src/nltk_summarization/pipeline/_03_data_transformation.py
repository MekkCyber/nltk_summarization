from nltk_summarization.config.configuration import ConfigurationManager
from nltk_summarization.components.data_transformation import DataTransformation
from nltk_summarization.logging import logger


class DataTransformationTrainingPipeline:
    def __init__(self):
        pass

    def main(self):
        config = ConfigurationManager()
        data_transformation_config = config.get_data_transformation_config()
        data_transformation = DataTransformation(config=data_transformation_config)
        data_transformation.convert()