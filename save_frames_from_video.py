"""
Utility script for creating a directory of frames from video files
"""
import cv2
import multiprocessing
from pathlib import Path

READ_EVERY_NTH_FRAME = 30 * 30
VIDEO_NAMES = [f'video{i}' for i in range(1, 9)]
VIDEO_NAMES.extend([f'video_cheating{i}' for i in range(1, 3)])


def save_frames_from_video(video_name):
    Path('./videos/' + video_name).mkdir(parents=True, exist_ok=True)
    cap = cv2.VideoCapture('./raw_videos/' + f'{video_name}.webm')

    cycle_counter = 0
    frame_counter = 0

    ret, frame = cap.read()
    while ret:
        cv2.imwrite(f'./videos/{video_name}/frame_{frame_counter}.jpg', frame)
        cycle_counter += READ_EVERY_NTH_FRAME
        frame_counter += 1
        cap.set(cv2.CAP_PROP_POS_FRAMES, cycle_counter)

        ret, frame = cap.read()

    cap.release()
    cv2.destroyAllWindows()


def main():
    with multiprocessing.Pool(5) as p:
        p.map(save_frames_from_video, VIDEO_NAMES)
        print('Done')


if __name__ == '__main__':
    main()
