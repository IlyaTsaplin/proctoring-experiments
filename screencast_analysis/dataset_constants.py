DATASET_PATH = "./video_data/dataset.csv"  # Path to stored dataset
VIDEOS_DIR_PATH = "./video_data/parsed_data"  # Path to parsed videos

# Templates for dataset columns for object classes
COUNTS_KEY = "{}_counts"
TEXT_LENGTHS_KEY = "{}_text_lengths"
TEXTS_KEY = "{}_texts"

# Columns not tied to an object class
VIDEO_LENGTH_KEY = "video_length"
IS_CHEATED_KEY = "is_cheated"

DATASET_COLUMNS = [COUNTS_KEY, TEXT_LENGTHS_KEY, TEXTS_KEY]

# Detected object classes
CLASSES = [
    "code_editor",
    "discord",
    "moodle",
    "notepad",
    "proctoring",
    "regex_editor",
    "search_engine",
    "terminal",
]
