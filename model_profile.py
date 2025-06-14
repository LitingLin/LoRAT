import os
from main import setup_arg_parser
from trackit.core.boot.main import main


if __name__ == '__main__':
    parser = setup_arg_parser()
    args = parser.parse_args()
    args.root_path = os.path.dirname(os.path.abspath(__file__))
    args.mixin_config = args.mixin_config if args.mixin_config else []
    args.mixin_config.append('do_model_profiling_only')
    args.dry_run = True
    main(args)
