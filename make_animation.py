from moviepy.video import VideoClip
from moviepy.video.io.ImageSequenceClip import ImageSequenceClip
import glob
from natsort import natsorted

def make_animation(spec_dir):
    images_list = glob.glob(spec_dir + '/*.png')
    images_list = natsorted(images_list, key=lambda y: y.lower())
    clip = ImageSequenceClip(images_list, fps=2)
    clip.write_videofile("spectrum_animation.mp4", audio=False)
