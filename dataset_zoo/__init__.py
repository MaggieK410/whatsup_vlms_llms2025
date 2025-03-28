from .aro_datasets import VG_Relation, VG_Attribution, COCO_Order, Flickr30k_Order, Controlled_Images, COCO_QA, VG_QA
from .retrieval import COCO_Retrieval, Flickr30k_Retrieval


def get_dataset(dataset_name, image_preprocess=None, text_perturb_fn=None, image_perturb_fn=None, download=False, split="test", *args, **kwargs):
    """
    Helper function that returns a dataset object with an evaluation function. 
    dataset_name: Name of the dataset.
    image_preprocess: Preprocessing function for images.
    text_perturb_fn: A function that takes in a string and returns a string. This is for perturbation experiments.
    image_perturb_fn: A function that takes in a PIL image and returns a PIL image. This is for perturbation experiments.
    download: Whether to allow downloading images if they are not found.
    """
    if dataset_name == "VG_Relation": 
        from .aro_datasets import get_visual_genome_relation
        return get_visual_genome_relation(image_preprocess=image_preprocess, text_perturb_fn=text_perturb_fn, image_perturb_fn=image_perturb_fn, download=download, *args, **kwargs)
    elif dataset_name == "VG_Attribution":
        from .aro_datasets import get_visual_genome_attribution
        return get_visual_genome_attribution(image_preprocess=image_preprocess, text_perturb_fn=text_perturb_fn, image_perturb_fn=image_perturb_fn, download=download, *args, **kwargs)
    elif dataset_name == "COCO_Order":
        from .aro_datasets import get_coco_order
        return get_coco_order(image_preprocess=image_preprocess, text_perturb_fn=text_perturb_fn, image_perturb_fn=image_perturb_fn, download=download, *args, **kwargs)
    elif dataset_name == "Flickr30k_Order":
        from .aro_datasets import get_flickr30k_order
        return get_flickr30k_order(image_preprocess=image_preprocess, text_perturb_fn=text_perturb_fn, image_perturb_fn=image_perturb_fn, download=download, *args, **kwargs)
    elif dataset_name == "Controlled_Images_A":
        from .aro_datasets import get_controlled_images_a
        return get_controlled_images_a(image_preprocess=image_preprocess, text_perturb_fn=text_perturb_fn, image_perturb_fn=image_perturb_fn, download=download, split=split, *args, **kwargs)
    elif dataset_name == "Controlled_Images_B":
        from .aro_datasets import get_controlled_images_b
        return get_controlled_images_b(image_preprocess=image_preprocess, text_perturb_fn=text_perturb_fn, image_perturb_fn=image_perturb_fn, download=download, *args, **kwargs)
    elif dataset_name == "left_right_dataset":
        from .aro_datasets import get_left_right_dataset
        return get_left_right_dataset(image_preprocess=image_preprocess, text_perturb_fn=text_perturb_fn,
                                       image_perturb_fn=image_perturb_fn, download=download, *args,
                                       **kwargs)
    elif dataset_name == "COCO_QA_one_obj":
        from .aro_datasets import get_coco_qa_one_obj
        return get_coco_qa_one_obj(image_preprocess=image_preprocess, text_perturb_fn=text_perturb_fn, image_perturb_fn=image_perturb_fn, download=download, *args, **kwargs)
    elif dataset_name == "COCO_QA_two_obj":
        from .aro_datasets import get_coco_qa_two_obj
        return get_coco_qa_two_obj(image_preprocess=image_preprocess, text_perturb_fn=text_perturb_fn, image_perturb_fn=image_perturb_fn, download=download, *args, **kwargs)
    elif dataset_name == "VG_QA_one_obj":
        from .aro_datasets import get_vg_qa_one_obj
        return get_vg_qa_one_obj(image_preprocess=image_preprocess, text_perturb_fn=text_perturb_fn, image_perturb_fn=image_perturb_fn, download=download, *args, **kwargs)
    elif dataset_name == "VG_QA_two_obj":
        from .aro_datasets import get_vg_qa_two_obj
        return get_vg_qa_two_obj(image_preprocess=image_preprocess, text_perturb_fn=text_perturb_fn, image_perturb_fn=image_perturb_fn, download=download, *args, **kwargs)
    elif dataset_name == "COCO_Retrieval":
        from .retrieval import get_coco_retrieval
        return get_coco_retrieval(image_preprocess=image_preprocess, text_perturb_fn=text_perturb_fn, image_perturb_fn=image_perturb_fn, download=download, *args, **kwargs)
    elif dataset_name == "Flickr30k_Retrieval":
        from .retrieval import get_flickr30k_retrieval
        return get_flickr30k_retrieval(image_preprocess=image_preprocess, text_perturb_fn=text_perturb_fn, image_perturb_fn=image_perturb_fn, download=download, *args, **kwargs)
    else:
        raise ValueError(f"Unknown dataset {dataset_name}")
