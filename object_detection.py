"""
Script for creating a time series for the amount of objects detected by the YOLO model from video 
"""
import glob
import pandas as pd
import numpy as np
from ultralytics import YOLO

VIDEO_NAMES = [f'video{i}' for i in range(1, 9)]
VIDEO_NAMES.extend([f'video_cheating{i}' for i in range(1, 3)])
PATH_TO_VIDEO_DIR = 'videos/'
PATH_TO_MODEL = 'best.pt'

model = YOLO(PATH_TO_MODEL)


def create_time_series(video_path: str):
    time_series = {name: [] for name in model.names.values()}

    for image_path in glob.glob(video_path + '*.jpg'):
        results = model.predict(image_path, imgsz=640, conf=0.25, iou=0.45)
        for result in results:
            entry = {name: 0 for name in result.names.values()}

            for cls in result.boxes.cls:
                entry[result.names[cls.item()]] += 1

            for name, value in entry.items():
                time_series[name].append(value)

    return time_series


def main():
    main_df = pd.DataFrame()
    for video_name in VIDEO_NAMES:
        time_series = create_time_series(PATH_TO_VIDEO_DIR + video_name + '/')


        data_array = []
        for _, series in time_series.items():
            data_array.append(series)
        data_array = np.array(data_array)
        print('Shape of data_array:', data_array.shape)

        df = pd.Series([data_array.reshape(-1)])
        df = df.to_frame()
        df.columns = ['values']
        df['cheating'] = 0 if 'cheating' not in video_name else 1

        main_df = pd.concat([main_df, df], ignore_index=True, axis=0)

    # Save to CSV
    main_df.to_csv(f'data.csv', index=False)


if __name__ == '__main__':
    main()
