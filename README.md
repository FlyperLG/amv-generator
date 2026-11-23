# amv-generator

yt-dlp -S "vcodec:h264,res:1080,acodec:m4a" --merge-output-format mp4 -o "data/amvs/%(title)s [%(id)s].%(ext)s" "<<url>>"