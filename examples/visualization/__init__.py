"""Offline 3DGS cutaway and prediction rendering."""


def render_result(*args, **kwargs):
    from .report import render_result as render

    return render(*args, **kwargs)
