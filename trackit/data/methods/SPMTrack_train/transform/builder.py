def build_transform(data_config, config, build_context, dtype):
    transform_config = data_config['transform']
    if transform_config['type'] == 'default':
        from .default.builder import build_SPMTrack_data_processing_components
        return build_SPMTrack_data_processing_components(transform_config, config, build_context, dtype)
    else:
        raise NotImplementedError('Unknown transform type: {}'.format(transform_config['type']))
