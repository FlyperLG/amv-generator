from pathlib import Path

from amv.io.ffmpeg import probe, run_proxy_transcode

out = probe("data/amvs/Agony Hardstyle (Blue Lock - Turn it into Light! ft. Yoichi Isagi Remix_AMV).mp4")
print(out)

run_proxy_transcode(in_path=out.path, out_dir=Path("data/proxies/"))
