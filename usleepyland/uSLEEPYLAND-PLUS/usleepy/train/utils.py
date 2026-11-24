"""Training helpers retained for loading and maintaining model code."""

import logging
import tensorflow
from typing import List

logger = logging.getLogger(__name__)

def ensure_list_or_tuple(obj):
    """
    Takes some object and wraps it in a list - i.e. [obj] - unless the object
    is already a list or a tuple instance. In that case, simply returns 'obj'

    Args:
        obj: Any object

    Returns:
        [obj] if obj is not a list or tuple, else obj
    """
    return [obj] if not isinstance(obj, (list, tuple)) else obj

def _get_classes_or_funcs(string_list: list, func_modules: list) -> List[callable]:
    """
    Helper for 'init_losses' or 'init_metrics'.
    Please refer to their docstrings.

    Args:
        string_list:  (list)   List of strings, each giving a name of a metric
                               or loss to use for training. The name should
                               refer to a function or class in either tf_funcs
                               or custom_funcs modules.
        func_modules: (module or list of modules) A Tensorflow.keras module of losses or metrics,
                                                  or a list of various modules to look through.

    Returns:
        A list of len(string_list) of classes/functions of losses/metrics/optimizers/activation functions etc.
    """
    functions_or_classes = []
    func_modules = ensure_list_or_tuple(func_modules)
    for func_or_class_str in ensure_list_or_tuple(string_list):
        found = False
        for module in func_modules:
            found = getattr(module, func_or_class_str, False)
            if found:
                logger.info(f"Found requested class '{func_or_class_str}' in module '{module}'")
                functions_or_classes.append(found)  # return the first found
                break
        if not found:
            raise AttributeError(f"Did not find loss/metric function {func_or_class_str} "
                                 f"in the module(s) '{func_modules}'")
    return functions_or_classes

def get_activation_function(activation_string):
    """
    Same as 'init_losses', but for optimizers.
    Please refer to the 'init_losses' docstring.
    """
    activation = _get_classes_or_funcs(
        activation_string,
        func_modules=[tensorflow.keras.activations]
    )
    assert len(activation) == 1, f'Received unexpected number of activation functions ({len(activation)}, expected 1)'
    return activation[0]
