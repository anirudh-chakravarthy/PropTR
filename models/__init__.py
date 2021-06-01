# from .detr import build
from .proptr import build


def build_model(args):
    return build(args)
