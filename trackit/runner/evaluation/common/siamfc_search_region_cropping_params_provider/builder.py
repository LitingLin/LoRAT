from timm.layers import to_2tuple

from trackit.miscellanies.pretty_format import pretty_format
from . import CroppingParameterProviderFactory


def build_siamfc_search_region_cropping_parameter_provider_factory(siamfc_search_region_cropping_config: dict):
    tracking_curation_process_method = siamfc_search_region_cropping_config['type']
    if tracking_curation_process_method == 'simple':
        from .simple import SiamFCCroppingParameterSimpleProvider
        min_object_size = siamfc_search_region_cropping_config.get('min_object_size', 1.)
        min_object_size = to_2tuple(min_object_size)
        area_factor = siamfc_search_region_cropping_config['area_factor']
        curation_parameter_provider = SiamFCCroppingParameterSimpleProvider(area_factor, min_object_size)
        print("Using simple SiamFC cropping strategy\n" + pretty_format(siamfc_search_region_cropping_config))
    else:
        raise NotImplementedError("Unknown curation parameter provider type: {}".format(tracking_curation_process_method))
    return CroppingParameterProviderFactory(curation_parameter_provider)
