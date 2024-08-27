from cartesia_mlx.models.lm import LM
from cartesia_mlx.utils.configure import sub_cfg


class Rene(LM):
    """Rene Language Model.

    Example:
        import mlx.core as mx
        import cartesia_mlx as cmx

        model = cmx.from_pretrained('cartesia-ai/Rene-v0.1-1.3b-4bit-mlx')
        model.set_dtype(mx.float32)

        prompt = "Rene Descartes was"

        print(prompt, end="", flush=True)
        for text in model.generate(
            prompt,
            max_tokens=500,
            eval_every_n=5,
            verbose=True,
            top_p=0.99,
            temperature=0.85,
        ):
            print(text, end="", flush=True)
    """

    base_cfg = sub_cfg(LM.base_cfg, _class_="models.rene.Rene")
