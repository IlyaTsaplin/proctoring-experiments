import pathlib

import numpy as np
import cv2

import video_processing
import detected_objects_aggregation
import transformer
import dataset_constants


# Analyzer parameters
COUNTS_PREDICTION_WEIGHT = 0.5
TEXT_LENGTHS_PREDICTION_WEIGHT = 0.5
COUNTS_MODEL = transformer.get_model(dataset_constants.COUNTS_KEY)
TEXT_LENGTHS_MODEL = transformer.get_model(dataset_constants.TEXT_LENGTHS_KEY)
READ_EVERY_NTH_FRAME = 30 * 30  # every 30 seconds assuming 30 fps video


def get_average_model_predictions(values: list[float], model) -> float:
    """
    Calculates the average model prediction over sliding windows of input values.

    Args:
        values (list[float]): A list of float values of input data.
        model: The trained model used to make predictions.

    Returns:
        float: The average prediction across all windows.
    """
    window_size = transformer.WINDOW_SIZE * len(dataset_constants.CLASSES)
    number_of_windows = len(values) // window_size
    if (len(values) % window_size) != 0:
        number_of_windows += 1

    average_prediction = 0
    for i in range(number_of_windows):
        window_values = values[
            i * window_size : min(len(values), (i + 1) * window_size)
        ]

        # Pad with zeros if the window is not full
        if len(window_values) < window_size:
            window_values.extend([0] * (window_size - len(window_values)))

        average_prediction += model.predict(np.array(window_values).reshape(1, -1))

    return average_prediction / number_of_windows


def analyze_video(video_path: pathlib.Path | cv2.VideoCapture) -> float:
    """
    Analyzes a screencast video to predict the likelihood of proctoring issues.

    Function extracts objects of interest from frames, applies aggregation and uses models to predict the likelihood of proctoring issues.

    Args:
        video_path (pathlib.Path): The path to the video file to be analyzed or cv2.VideoCapture object.

    Returns:
        float: The predicted likelihood of proctoring issues, ranging from 0 to 1.
    """

    object_detection_result = video_processing.extract_objects_from_video(
        video_path, READ_EVERY_NTH_FRAME
    )

    aggregated_data_dict = detected_objects_aggregation.get_aggregated_data(
        object_detection_result
    )

    counts = []
    text_lengths = []
    for aggregated_data in aggregated_data_dict.values():
        counts.extend(aggregated_data.counts)
        text_lengths.extend(aggregated_data.text_lengths)

    average_counts_prediction = get_average_model_predictions(counts, COUNTS_MODEL)
    average_text_lengths_prediction = get_average_model_predictions(
        text_lengths, TEXT_LENGTHS_MODEL
    )

    return (
        COUNTS_PREDICTION_WEIGHT * average_counts_prediction
        + TEXT_LENGTHS_PREDICTION_WEIGHT * average_text_lengths_prediction
    )[0][1]


def demo():
    video_path = pathlib.Path("./video_data/raw_data/video1.webm")
    print(analyze_video(video_path))


if __name__ == "__main__":
    demo()
