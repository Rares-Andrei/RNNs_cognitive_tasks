import torch


def generate_temporal_xor(size: int, batch_size: int,
                          device=None, dtype=torch.float32):
    """
    Generates sequence where every 3rd element is the XOR of the previous two.

    Args:
        size (int): Size of sequence
        batch_size (int): Batch size for training
        device: Which device to use
        dtype: What data type it should return

    Return:
        sequence (torch.Tensor): Shape (size, batch_size, 1)
    """
    assert isinstance(size, int)
    assert isinstance(batch_size, int)
    sequence = torch.zeros(size=(size, batch_size, 1),
                           device=device, dtype=dtype)
    # I need to iterate through batches and do the filling
    for batch in range(batch_size):
        for index in range(1, size + 1):
            if index % 3 != 0:
                sequence[index-1, batch, 0] = torch.randint(low=0, high=2, size=(1,),
                                                            device=device).to(dtype)
            else:
                sequence[index-1, batch, 0] = ((sequence[index - 2, batch] !=
                                               sequence[index - 3, batch])).to(dtype)
    return sequence


def temporal_xor_nextstep_batch(seq_len: int, batch_size: int,
                                device=None, dtype=torch.float32):
    """
    Generate an iterator object to train the model on.

    Args:
        seq_len: How long the sequence is    
        batch_size: Size of the batch
        device: Which device to use
        dtype: What data type it should return

    """
    while True:
        seq = generate_temporal_xor(size=seq_len, batch_size=batch_size,
                                    device=device, dtype=dtype)
        x = seq[:-1]  # Input until penultimate element
        y = seq[1:]  # Labels starting from first element
        yield x, y
