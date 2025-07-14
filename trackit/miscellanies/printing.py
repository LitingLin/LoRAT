def print_centered_text(text: str, width: int = 80, fill_char: str = u'\u2500') -> None:
    """
    Print text centered in a line of given width, padded with a specified character.

    Args:
        text (str): The text to print.
        width (int): The total line width.
        fill_char (str): The character used for padding (defaults to u'\u2500'(looks like '─')).
    """
    text_length = len(text)
    if text_length >= width:
        # If the text is longer than or equal to the line width, just print it as is.
        print(text)
        return

    left_padding_length = (width - text_length) // 2
    right_padding_length = width - text_length - left_padding_length

    left_padding = fill_char * left_padding_length
    right_padding = fill_char * right_padding_length
    print(left_padding + text + right_padding)


def indent_format(string, indent_level: int = 0):
    if indent_level == 0:
        return string
    indent_str = '  ' * indent_level  # two spaces per level

    # Split the incoming message on newlines
    lines = string.split('\n')

    # Add the prefix to each line
    prefixed_lines = [f"{indent_str}{line}" for line in lines]

    return '\n'.join(prefixed_lines)


def pretty_format(object_, indent_level=0):
    import yaml
    try:
        from yaml import CSafeDumper as Dumper
    except ImportError:
        from yaml import SafeDumper as Dumper

    string = yaml.dump(object_, Dumper=Dumper, indent=2, default_flow_style=False, allow_unicode=True, sort_keys=False).strip()
    return indent_format(string, indent_level)
