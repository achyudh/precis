import abc


class OutputProcessor(abc.ABC):
    """
    A model-specific instance of the inherited class is passed to the classes that use its
    output in order to process it into the format that the class expects to receive it.
    """
    @abc.abstractmethod
    def process(self, *args, **kwargs):
        pass
