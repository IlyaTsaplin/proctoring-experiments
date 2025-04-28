import pathlib
import glob
import dataclasses
import pytesseract
import torch
import cv2

from PIL import Image
from ultralytics import YOLO

# Path to the trained model
YOLO_OBJECT_DETECTOR = "./model_weights/yolo.pt"
# Load the YOLO model
model = YOLO(YOLO_OBJECT_DETECTOR)


OBJECT_TO_LANGUAGE_MAP = {
    "code_editor": "eng",
    "regex_editor": "eng",
    "terminal": "eng",
}
DEFAULT_LANGUAGE = "rus"


@dataclasses.dataclass
class DetectedObject:
    """Dataclass for storing detected objects from frame."""

    class_name: str
    text: str
    size: int


@dataclasses.dataclass
class ObjectExtractionResult:
    """Dataclass for storing detected objects from video."""

    objects_per_frame: list[list[DetectedObject]]
    names: list[str]


def preprocess_image(image: Image.Image) -> Image.Image:
    """
    Convert an image to grayscale and apply binary thresholding.

    Args:
        image (pathlib.Path): The path to the image file.

    Returns:
        Image: The processed PIL Image.
    """
    gray_image = image.convert("L")
    binary_image = gray_image.point(lambda p: 255 if p > 128 else 0)
    return binary_image


def detect_text(
    preprocessed_image: Image, box_coordinates: torch.Tensor, object_name: str = ""
) -> str:
    """
    Detect text within given box coordinates of the image.

    Args:
        preprocessed_image (Image): Image to process for text detection.
        box_coordinates (torch.Tensor): Bounding box coordinates for cropping.

    Returns:
        str: The detected text.
    """
    # Convert box_coordinates to crop parameters
    left, top, right, bottom = box_coordinates

    # Crop the image according to the bounding box
    cropped_image = preprocessed_image.crop(
        (left.item(), top.item(), right.item(), bottom.item())
    )

    # Perform OCR on the cropped image
    extracted_text = pytesseract.image_to_string(
        cropped_image, lang=OBJECT_TO_LANGUAGE_MAP.get(object_name, DEFAULT_LANGUAGE)
    )

    return extracted_text


def extract_objects_from_frames(
    frames: list[Image.Image],
) -> ObjectExtractionResult:
    """
    Process a video frames to detect objects and their text.

    Args:
        frames (list[Image.Image]): List of frames

    Returns:
        ObjectExtractionResult: Nested list of detected objects with their text for each frame.
    """
    model_results_per_frame = model.predict(frames, imgsz=640, conf=0.25, iou=0.45)
    if len(model_results_per_frame) == 0:
        return ObjectExtractionResult(objects_per_frame=[], names=[])

    detected_objects_per_frame = []
    for model_results, frame in zip(model_results_per_frame, frames):
        preprocessed_image = preprocess_image(frame)
        detected_objects = []

        all_coordinates = model_results.boxes.xyxy
        cls_labels = model_results.boxes.cls

        for cls_label, coordinates in zip(cls_labels, all_coordinates):
            class_name = model_results.names[cls_label.item()]
            detected_objects.append(
                DetectedObject(
                    class_name=class_name,
                    text=detect_text(preprocessed_image, coordinates, class_name),
                    size=(coordinates[2] - coordinates[0])
                    * (coordinates[3] - coordinates[1]),
                )
            )

        detected_objects_per_frame.append(detected_objects)

    return ObjectExtractionResult(
        objects_per_frame=detected_objects_per_frame,
        names=[model_results.names[i] for i in range(len(model_results.names))],
    )


def extract_objects_from_parsed_video(
    video_path: pathlib.Path,
) -> ObjectExtractionResult:
    """
    Extract detected objects from each frame of a video.

    Args:
        video_path (pathlib.Path): The path to the directory containing video frame images.

    Returns:
        ObjectExtractionResult: A list where each element is a list of DetectedObject instances
        representing objects detected in each frame.
    """

    image_paths = [
        pathlib.Path(path_str) for path_str in glob.glob(str(video_path / "*.jpg"))
    ]
    return extract_objects_from_frames(
        [Image.open(image_path).copy() for image_path in image_paths]
    )


def extract_objects_from_video(
    video: pathlib.Path | cv2.VideoCapture,
    step: int = 1,
) -> ObjectExtractionResult:
    """
    Extract detected objects from each frame of a video.

    Args:
        video (pathlib.Path | cv2.VideoCapture): The path to the video file.

    Returns:
        ObjectExtractionResult: A list where each element is a list of DetectedObject instances
        representing objects detected in each frame.
    """
    if not isinstance(video, cv2.VideoCapture):
        video = cv2.VideoCapture(video)
    frames = []
    frame_index = 0
    while video.isOpened():
        success, frame = video.read()
        if not success:
            break
        frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        frames.append(Image.fromarray(frame))

        frame_index += step
        video.set(cv2.CAP_PROP_POS_FRAMES, frame_index)

    video.release()

    return extract_objects_from_frames(frames)
