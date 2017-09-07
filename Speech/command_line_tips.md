# Commandline a/v analysis tips

### Check information about video file
[source](https://trac.ffmpeg.org/wiki/FFprobeTips)
```shell
# ffprobe is a simple multimedia streams analyzer
ffprobe -v quiet -of csv=p=0 -show_entries format=duration video.mp4 # e.g. 2.34 (sec)
```

### Merge video with audio
[source](https://superuser.com/questions/277642/how-to-merge-audio-and-video-file-in-ffmpeg)
```shell
ffmpeg -i video.mp4 -i audio.wav out.mp4
ffmpeg -i video.mp4 -i audio.wav -shortest out.mp4
ffmpeg -i video.mp4 -i audio.wav -c:v copy -c:a aac -strict experimental out.mp4
```

### Convert .mp4 to .avi
[source](https://askubuntu.com/questions/83161/use-ffmpeg-to-transform-mp4-to-same-high-quality-avi-file)
```shell
ffmpeg -i video.mp4 -qscale 0 out.avi
```