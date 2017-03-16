
def flatten_losses(losses):
    return list(set(_recursive_flatten(losses)))


def _recursive_flatten(losses):
    # flatten categories
    if isinstance(losses, dict):
        yield from map(_recursive_flatten, losses.values())

    # flatten tower
    elif isinstance(losses, list):
        yield from map(_recursive_flatten, losses)

    # extract tensor
    elif isinstance(losses, tuple):
        device, loss = losses
        yield loss
