"""LingBotNavi: one Spatial Memory, one question."""

__all__ = ["LingBotNavi"]


def __getattr__(name):
    if name == "LingBotNavi":
        from .navigation_transformer import LingBotNavi

        return LingBotNavi
    raise AttributeError(name)
