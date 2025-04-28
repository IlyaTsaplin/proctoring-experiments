import dataclasses
import video_processing


@dataclasses.dataclass
class AggregatedData:
    counts: list[int] = dataclasses.field(default_factory=list)
    text_lengths: list[int] = dataclasses.field(default_factory=list)
    sizes: list[int] = dataclasses.field(default_factory=list)
    texts: list[str] = dataclasses.field(default_factory=list)


def get_aggregated_data(
    extracted_data: video_processing.ObjectExtractionResult,
) -> dict[str, AggregatedData]:
    aggregated_data_dict = {name: AggregatedData() for name in extracted_data.names}

    for frame_objects in extracted_data.objects_per_frame:
        for aggregated_data in aggregated_data_dict.values():
            aggregated_data.counts.append(0)
            aggregated_data.text_lengths.append(0)
            aggregated_data.sizes.append(0)
            aggregated_data.texts.append("")

        for detected_object in frame_objects:
            aggregated_data_dict[detected_object.class_name].counts[-1] += 1
            aggregated_data_dict[detected_object.class_name].text_lengths[-1] += len(
                detected_object.text
            )
            aggregated_data_dict[detected_object.class_name].sizes[-1] += (
                detected_object.size
            )
            aggregated_data_dict[detected_object.class_name].texts[-1] += (
                detected_object.text + "\n"
            )

    return aggregated_data_dict
