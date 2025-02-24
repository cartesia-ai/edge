from cartesia_mlx.models.lm import LM
from cartesia_mlx.utils.configure import sub_cfg


class Mohawk(LM):
    """Mohawk Language Model.

    Example:
        import mlx.core as mx
        import cartesia_mlx as cmx

        model = cmx.from_pretrained('cartesia-ai/Mohawk-v0.1-1.3B-4bit-mlx')
        model.set_dtype(mx.float32)

        prompt = "Once upon a time in a land far, far away,"

        print(prompt, end="", flush=True)
        for text in model.generate(prompt, temp=1.0, max_tokens=1000, eval_every_n=1):
            print(text, end="", flush=True)
    """

    base_cfg = sub_cfg(LM.base_cfg, _class_="models.mohawk.Mohawk")
