import abc

from lib.data import PathContextInput


class AbstractInputProcessor(abc.ABC):
    """
    A model-specific instance of the inherited class is passed to the reader in order to help it
    construct the input in the form that the model expects to receive it.
    """

    @abc.abstractmethod
    def process(self, input_tensors: PathContextInput):
        pass
