from trackit.models.schema.data_schema import get_model_input_output_data_schema, ModelInputOutputDataSchema


def auto_unpack_and_call(data, model):
    data_schema = get_model_input_output_data_schema(data)
    if data_schema == ModelInputOutputDataSchema.Singleton:
        return model(data)
    elif data_schema == ModelInputOutputDataSchema.List:
        return model(*data)
    elif data_schema == ModelInputOutputDataSchema.Dict:
        return model(**data)
    else:
        raise ValueError('Unsupported data schema')
