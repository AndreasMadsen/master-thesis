
import numpy as np


def size_to_signed_type(size: int) -> np.signedinteger:
    for np_type in [np.int8, np.int16, np.int32, np.int64]:
        if size <= np.iinfo(np_type).max:
            return np_type


def size_to_unsigned_type(size: int) -> np.unsignedinteger:
    for np_type in [np.uint8, np.uint16, np.uint32, np.uint64]:
        if size <= np.iinfo(np_type).max:
            return np_type
