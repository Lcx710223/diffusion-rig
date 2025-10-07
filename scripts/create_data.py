###LCX250928,COPILOT修改。LCX251007修改BATCH-SIZE=1。
import os, sys
from tqdm import tqdm
import torch as th

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
from decalib.deca import DECA
from decalib.datasets import datasets
from decalib.utils.config import cfg as deca_cfg

import pickle
from torch.utils.data import DataLoader
import lmdb
from PIL import Image
from io import BytesIO
import pickle
import argparse
import torch
from utils.script_util import (
    add_dict_to_argparser,
)


def main():
    args = create_argparser().parse_args()
    device = "cuda" if torch.cuda.is_available() else "cpu"

    # Build DECA
    deca_cfg.model.use_tex = True
    deca_cfg.model.tex_path = "data/FLAME_texture.npz"
    deca_cfg.model.tex_type = "FLAME"
    ### LCX,CP,强调兼容性。 deca = DECA(config=deca_cfg, device="cuda")
    device = "cuda" if torch.cuda.is_available() else "cpu"
    deca = DECA(config=deca_cfg, device=device)

    # Create Dataset
    dataset_root = args.data_dir
    testdata = datasets.TestData(
        dataset_root, iscrop=True, size=args.image_size, sort=True
    )
    # 插入调试语句，打印加载图像数量
    print(f"LCX: TestData 加载图像数量: {len(testdata)}")
    batch_size = args.batch_size
    loader = DataLoader(testdata, batch_size=batch_size)

    output_dir = args.output_dir
    os.makedirs(output_dir, exist_ok=True)

    if args.use_meanshape:
        shapes = []
        for td in testdata:
            img = td["image"].to(device).unsqueeze(0)
            code = deca.encode(img)
            shapes.append(code["shape"].detach())
        mean_shape = th.mean(th.cat(shapes, dim=0), dim=0, keepdim=True)

        with open(os.path.join(output_dir, "mean_shape.pkl"), "wb") as f:
            pickle.dump(mean_shape, f)

    with lmdb.open(output_dir, map_size=1024**4, readahead=False) as env:

        total = 0
        for batch_id, data in enumerate(tqdm(loader)):
            print(f"LCX:写入帧编号: {batch_id}")  #LCX: 插入这里
            
            with th.no_grad():
                inp = data["image"].to(device)
                codedict = deca.encode(inp)
                tform = data["tform"]
                tform = th.inverse(tform).transpose(1, 2).to(device)
                original_image = data["original_image"].to(device)

                if args.use_meanshape:
                    codedict["shape"] = mean_shape.repeat(inp.shape[0], 1)
                codedict["tform"] = tform

                opdict, _ = deca.decode(
                    codedict,
                    render_orig=True,
                    original_image=original_image,
                    tform=tform,
                )
                opdict["inputs"] = original_image

                for item_id in range(inp.shape[0]):
                    i = batch_id * batch_size + item_id

                    image = (
                        (original_image[item_id].detach().cpu().numpy() * 255)
                        .astype("uint8")
                        .transpose((1, 2, 0))
                    )
                    image = Image.fromarray(image)

                    albedo_key = f"albedo_{str(i).zfill(6)}".encode("utf-8")
                    buffer = BytesIO()
                    pickle.dump(opdict["albedo_images"][item_id].detach().cpu(), buffer)
                    albedo_val = buffer.getvalue()

                    normal_key = f"normal_{str(i).zfill(6)}".encode("utf-8")
                    buffer = BytesIO()
                    pickle.dump(opdict["normal_images"][item_id].detach().cpu(), buffer)
                    normal_val = buffer.getvalue()

                    rendered_key = f"rendered_{str(i).zfill(6)}".encode("utf-8")
                    buffer = BytesIO()
                    pickle.dump(
                        opdict["rendered_images"][item_id].detach().cpu(), buffer
                    )
                    rendered_val = buffer.getvalue()

                    image_key = f"image_{str(i).zfill(6)}".encode("utf-8")
                    buffer = BytesIO()
                    image.save(buffer, format="png", quality=100)
                    image_val = buffer.getvalue()

                    with env.begin(write=True) as transaction:
                        transaction.put(albedo_key, albedo_val)
                        transaction.put(normal_key, normal_val)
                        transaction.put(rendered_key, rendered_val)
                        transaction.put(image_key, image_val)
                    total += 1
        with env.begin(write=True) as transaction:
            transaction.put("length".encode("utf-8"), str(total).encode("utf-8"))
        print(f"LCX : 最终写入帧数: {total}") ###LCX

def create_argparser():
    defaults = dict(
        data_dir="",
        output_dir="",
        image_size=256,
        batch_size=1, ###LCX251007。
        use_meanshape=False,
    )
    parser = argparse.ArgumentParser()
    add_dict_to_argparser(parser, defaults)
    return parser


if __name__ == "__main__":
    main()
