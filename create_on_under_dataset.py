import os
import shutil
import json

src_dirs = [
    "/mnt/c/Users/nayou/Personal/MVA/llm/whatsup_vlms_llms2025/data/controlled_clevr",
    "/mnt/c/Users/nayou/Personal/MVA/llm/whatsup_vlms_llms2025/data/controlled_images"
]

dst_dir = "/mnt/c/Users/nayou/Personal/MVA/llm/whatsup_vlms_llms2025/data/on_under_images"
os.makedirs(dst_dir, exist_ok=True)

data = []

for src in src_dirs:
    for file in os.listdir(src):
        if file.endswith(".jpeg") and any(x in file for x in ["_on_", "_under_"]):
            src_path = os.path.join(src, file)
            dst_path = os.path.join(dst_dir, file)
            shutil.copy(src_path, dst_path)

            relation = "on" if "_on_" in file else "under"
            opposite = "under" if relation == "on" else "on"
        
            description = file.replace("_", " ").replace(".jpeg", "")
            
            caption_options = [
                description,
                description.replace(relation, opposite)
            ]
            
            data.append({
                "image_path": os.path.join("data/on_under_images", file),
                "caption_options": caption_options
            })

json_path = os.path.join(dst_dir, "on_under_images.json")
with open(json_path, "w") as f:
    json.dump(data, f, indent=4)
