from datetime import datetime
import os


def get_datetime_format():
    return "%Y.%m.%d-%H.%M.%S-%f"


def generate_run_id(args, extra_parts=()):
    parts = [args.method_name, args.config_name, *extra_parts]
    if args.mixin_config is not None:
        for mixin_config in args.mixin_config:
            parts.append('mixin')
            parts.append(os.path.basename(mixin_config))
    parts.append(datetime.now().strftime(get_datetime_format()))

    return '-'.join(parts)
