import cv2
import imageio

def compile_to_mp4(frames, fps, path):
    height, width, layers = frames[0].shape

    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out = cv2.VideoWriter(path, fourcc, fps, (width, height))

    for frame in frames:
        out.write(frame)

    out.release()

def compile_to_gif(frames, fps, path):
    duration = 1 / fps 
    imageio.mimsave(path, frames, format='GIF', duration=duration)
    

