def build_box_with_score_map_label_generator(config: dict):
    common_config = config['common']
    from . import BoxWithScoreMapLabelGenerator, box_with_score_map_label_collator
    return (BoxWithScoreMapLabelGenerator(common_config['response_map_size'], common_config['search_region_size']),
            box_with_score_map_label_collator)
