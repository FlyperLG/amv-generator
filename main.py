from pathlib import Path

from tqdm import tqdm

from amv.io.ffmpeg import iter_frames, make_audio_proxy, make_video_proxy, probe

out = probe("data/amvs/Agony Hardstyle (Blue Lock - Turn it into Light! ft. Yoichi Isagi Remix_AMV).mp4")
print(out)

proxy_path = make_video_proxy(in_path=out.path, out_dir=Path("data/proxies/"))

fps = 30
x = iter_frames(proxy_path, fps, 640, 360, 0, None, True)

for _ in tqdm(x, total=out.video.duration * fps):
    continue

make_audio_proxy(out.path, Path("data/proxies/"))
