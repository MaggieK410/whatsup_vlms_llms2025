import os
import shutil
import json

src_dirs = [
    "/Users/itazaporozhets/VSCode/whatsup_vlms_llms2025/data/controlled_clevr",
    "/Users/itazaporozhets/VSCode/whatsup_vlms_llms2025/data/controlled_images"
]
dst_dir = "/Users/itazaporozhets/VSCode/whatsup_vlms_llms2025/data/left_right_images"
os.makedirs(dst_dir, exist_ok=True)

data = []

for src in src_dirs:
    for file in os.listdir(src):
        if any(x in file for x in ["left", "right"]) and file.endswith(".jpeg"):
            src_path = os.path.join(src, file)
            dst_path = os.path.join(dst_dir, file)
            shutil.copy(src_path, dst_path)

            position = "left" if "left" in file else "right"
            opposite = "right" if position == "left" else "left"
            description = file.replace("_", " ").replace(".jpeg", "")
            caption_options = [
                f"{description.replace(position, position)}",
                f"{description.replace(position, opposite)}"
            ]

            data.append(
                {"image_path": os.path.join("data/left_right_images", file), "caption_options": caption_options})

json_path = os.path.join(dst_dir, "left_right_images.json")
with open(json_path, "w") as f:
    json.dump(data, f, indent=4)
