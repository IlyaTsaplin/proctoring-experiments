import pathlib
import typing

from matplotlib import pyplot as plt

import video_processing
import detected_objects_aggregation


VIDEOS_PATH = "./video_data/parsed_data/video1"


def plot_data(
    aggregated_data_dict,
    values_extractor: typing.Callable[
        [detected_objects_aggregation.AggregatedData], list[int]
    ],
    title="",
):
    plt.figure(figsize=(8, 6))

    for label, aggregated_data in aggregated_data_dict.items():
        values = values_extractor(aggregated_data)
        plt.plot(range(len(values)), values, label=label)
    plt.legend()

    if title:
        plt.title(title)
    plt.show()


def main():
    """
    Demo script that processes a single video and plots object counts and text lengths
    """

    object_detection_result = video_processing.extract_objects_from_video_path(
        pathlib.Path(VIDEOS_PATH)
    )

    aggregated_data_dict = detected_objects_aggregation.get_aggregated_data(
        object_detection_result
    )

    plot_data(
        aggregated_data_dict,
        lambda data: data.sizes,
        "Sizes",
    )
    plot_data(
        aggregated_data_dict,
        lambda data: data.counts,
        "Counts",
    )
    plot_data(
        aggregated_data_dict,
        lambda data: data.text_lengths,
        "Text lengths",
    )


if __name__ == "__main__":
    main()
