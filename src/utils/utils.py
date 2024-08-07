import cv2
import imageio
import numpy as np


def compile_to_mp4(frames, fps, path):
    height, width, layers = frames[0].shape

    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out = cv2.VideoWriter(path, fourcc, fps, (width, height))

    for i, frame in enumerate(frames):
        out.write(frame)

    out.release()
    
if __name__ == "__main__":
    import time
    import numpy as np
    
    
    frames = np.random.randint(0, 255, size=(1000, 320, 320, 3))
    
    mp4_start_time = time.perf_counter()
    compile_to_mp4(frames=frames, fps=30, path='video.mp4')
    mp4_end_time = time.perf_counter() - mp4_start_time
    
    print(f'compiled {frames.shape} to mp4 in {round(mp4_end_time, 2)}')