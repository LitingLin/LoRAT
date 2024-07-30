def print_centered_text(text: str, total_line_length: int = 80, padding_char: str = u'\u2500'):
    left_padding_length = (total_line_length - len(text)) // 2
    if left_padding_length < 0:
        left_padding_length = 0
    right_padding_length = total_line_length - len(text) - left_padding_length
    if right_padding_length < 0:
        right_padding_length = 0
    print(padding_char * left_padding_length + text + padding_char * right_padding_length)
