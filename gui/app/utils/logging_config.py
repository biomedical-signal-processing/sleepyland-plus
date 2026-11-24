"""Configure diagnostic logging for the Flask web interface."""

import logging  # Python's standard logging module for configurable log output


def configure_logging():
    """
    Configure and return a logger for the current module.

    This function:
      - Sets up the global logging configuration (once) using basicConfig.
      - Defines:
          * The minimum log level to record (DEBUG, so you see everything).
          * The format of each log message (timestamp, level, and message).
      - Returns a logger instance bound to the current module's name (__name__),
        which you can use throughout your code with log.debug(), log.info(), etc.

    Returns
    -------
    logging.Logger
        A logger configured with the specified level and format.
    """
    # Configure the root logger:
    # - level=logging.DEBUG: capture DEBUG and above (INFO, WARNING, ERROR, CRITICAL)
    # - format: include timestamp, log level, and the actual message
    logging.basicConfig(
        level=logging.DEBUG,
        format='%(asctime)s %(levelname)s: %(message)s'
    )

    # Return a logger specific to this module's namespace.
    # In other files, you'll typically call this once and assign:
    #     log = configure_logging()
    # Then use:
    #     log.debug("Debug message")
    #     log.info("Info message")
    #     log.error("Error message")
    return logging.getLogger(__name__)
