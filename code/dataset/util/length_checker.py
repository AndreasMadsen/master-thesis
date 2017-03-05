
import abc


class LengthChecker:
    _min_length: int
    _max_length: int

    def __new__(cls, min_length: int=None, max_length: int=None):
        if min_length is not None and max_length is not None:
            return LengthCheckerMinMax(min_length, max_length)
        elif min_length is not None:
            return LengthCheckerMin(min_length, max_length)
        elif max_length is not None:
            return LengthCheckerMax(min_length, max_length)
        else:
            return LengthCheckerNoop(min_length, max_length)

    def __init__(self, min_length: int=None, max_length: int=None):
        self._min_length = min_length
        self._max_length = max_length

    @abc.abstractmethod
    def __call__(self, source: str, target: str):
        pass


class LengthCheckerSpecialization(LengthChecker):
    def __new__(cls, *args, **kwargs):
        # this will implicitly call __init__ because an cls instance
        # is returned
        return object.__new__(cls)


class LengthCheckerMinMax(LengthCheckerSpecialization):
    def __call__(self, source: str, target: str):
        return self._min_length <= len(source) < self._max_length and \
               self._min_length <= len(target) < self._max_length


class LengthCheckerMin(LengthCheckerSpecialization):
    def __call__(self, source: str, target: str):
        return self._min_length <= len(source) and \
               self._min_length <= len(target)


class LengthCheckerMax(LengthCheckerSpecialization):
    def __call__(self, source: str, target: str):
        return len(source) < self._max_length and \
               len(target) < self._max_length


class LengthCheckerNoop(LengthCheckerSpecialization):
    def __call__(self, source: str, target: str):
        return True
