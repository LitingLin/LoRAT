def build_transform(data_config, config, build_context):
    transform_config = data_config['transform']
    if transform_config['type'] == 'default':
        from .default.builder import build_siamese_tracker_training_data_processing_components
        return build_siamese_tracker_training_data_processing_components(transform_config, config, build_context)
    else:
        raise NotImplementedError('Unknown transform type: {}'.format(transform_config['type']))
