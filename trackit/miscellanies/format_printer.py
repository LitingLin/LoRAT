# https://stackoverflow.com/questions/44356693/pprint-with-custom-float-formats
import pprint
class FormatPrinter(pprint.PrettyPrinter):
    def __init__(self, formats):
        super(FormatPrinter, self).__init__()
        self.formats = formats

    def format(self, obj, ctx, maxlvl, lvl):
        if type(obj) in self.formats:
            return self.formats[type(obj)] % obj, 1, 0
        return pprint.PrettyPrinter.format(self, obj, ctx, maxlvl, lvl)
