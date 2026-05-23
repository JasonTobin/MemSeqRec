import os
import sys
import warnings

# Must be set before any TensorFlow / Keras import.
# tf.compat.v1.layers.* is not available under standalone Keras 3;
# this tells TF to use its bundled legacy Keras 2 instead.
os.environ.setdefault('TF_USE_LEGACY_KERAS', '1')

from au2actr import Au2ActrError
from au2actr.commands import create_argument_parser
from au2actr.configuration import load_configuration
from au2actr.logging import get_logger, enable_verbose_logging


def main(argv):
    try:
        parser = create_argument_parser()
        arguments = parser.parse_args(argv[1:])
        if arguments.verbose:
            enable_verbose_logging()
        if arguments.command == 'train':
            from au2actr.commands.train import entrypoint
        elif arguments.command == 'eval':
            from au2actr.commands.eval import entrypoint
        else:
            raise Au2ActrError(
                f'pisa does not support commands {arguments.command}')
        params = load_configuration(arguments.configuration)
        entrypoint(params)
    except Au2ActrError as e:
        get_logger().error(e)


def entrypoint():
    """ Command line entrypoint. """
    warnings.filterwarnings('ignore')
    main(sys.argv)


if __name__ == '__main__':
    entrypoint()
