import argparse

parser = argparse.ArgumentParser()
parser.add_argument(
    "--dataset", type=str, default=R"D:\datasets\GIDAS_V3_clean", help="The path to the data set."
)
parser.add_argument(
    "--model_path",
    type=str,
    default="./ckpt/pddn.pth",
    help="The path to the model checkpoint.",
)
parser.add_argument(
    "--backbone",
    type=str,
    default=R"D:\checkpoints\sam_vit_h_4b8939.pth",
    help="The path to the backbone checkpoint.",
)
parser.add_argument(
    "--type",
    type=str,
    default="vit_h",
    choices=["vit_b", "vit_l", "vit_h"],
    help="The path to the model checkpoint.",
)
parser.add_argument(
    "--device", type=str, default="cuda:0", help="model compute device, cpu or cuda"
)
parser.add_argument(
    "--part",
    type=str,
    default="val",
    choices=["train", "test", "val"],
    help="Which part of the FSC147 data needs to be used.",
)
parser.add_argument("--log_dir", type=str, default="../log/", help="Path to output logs.")
