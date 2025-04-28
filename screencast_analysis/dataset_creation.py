import multiprocessing
import pathlib
import glob

import pandas as pd

import video_processing as video_processing
import detected_objects_aggregation as detected_objects_aggregation
import screencast_analysis.dataset_constants as dataset_constants

DATASET_PATH = "./video_data/dataset.csv"
PROCESSOR_POOL_SIZE = 5


def create_dataset():
    video_paths = glob.glob(str(pathlib.Path(dataset_constants.VIDEOS_DIR_PATH) / "*"))
    with multiprocessing.Pool(PROCESSOR_POOL_SIZE) as video_processing_pool:
        object_detection_results = video_processing_pool.map(
            video_processing.extract_objects_from_video_path,
            [pathlib.Path(path) for path in video_paths],
        )

    data_frame = None
    for object_detection_result, video_path in zip(
        object_detection_results, video_paths
    ):
        aggregated_data_dict = detected_objects_aggregation.get_aggregated_data(
            object_detection_result
        )

        if data_frame is None:
            columns = []
            for dataset_column in dataset_constants.DATASET_COLUMNS:
                columns.extend(
                    dataset_column.format(key) for key in aggregated_data_dict.keys()
                )
            columns.append(dataset_constants.IS_CHEATED_KEY)
            data_frame = pd.DataFrame(columns=columns)

        data_row = {}
        for class_name, aggregated_data in aggregated_data_dict.items():
            data_row[dataset_constants.COUNTS_KEY.format(class_name)] = [
                aggregated_data.counts
            ]
            data_row[dataset_constants.TEXT_LENGTHS_KEY.format(class_name)] = [
                aggregated_data.text_lengths
            ]
            data_row[dataset_constants.TEXTS_KEY.format(class_name)] = [
                aggregated_data.texts
            ]

        data_row[dataset_constants.IS_CHEATED_KEY] = ["cheating" in video_path]
        data_row[dataset_constants.VIDEO_LENGTH_KEY] = [len(aggregated_data.counts)]

        data_frame = pd.concat([data_frame, pd.DataFrame(data_row)])

    if data_frame is not None:
        data_frame.to_csv(DATASET_PATH)


if __name__ == "__main__":
    create_dataset()
