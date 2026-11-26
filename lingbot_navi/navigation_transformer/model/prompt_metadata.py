def image_bounds(ids, *, vision_start, vision_end, image_token):
    """Match current_image_mask_from_ids for one already-unpadded CPU row."""
    if vision_start not in ids:
        if image_token in ids or vision_end in ids:
            raise ValueError("prompt has no current image")
        return (0, 0)
    start = ids.index(vision_start)
    end = next((i for i in range(start + 1, len(ids)) if ids[i] == vision_end), None)
    if end is None:
        raise ValueError("current image has no vision_end")
    if end <= start + 1:
        raise ValueError("invalid current image span")
    if any((value != image_token for value in ids[start + 1 : end])):
        raise ValueError("first vision span must contain only expanded image tokens")
    return (start, end + 1)
