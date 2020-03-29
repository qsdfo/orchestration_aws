"""
Defines Exceptions needed when parsing LSDB leadsheets
"""


class TieException(Exception):
    def __init__(self, value):
        self.value = value

    def __str__(self):
        return repr(self.value)


class ParsingException(Exception):
    def __init__(self, value):
        self.value = value

    def __str__(self):
        return repr(self.value)


class LeadsheetParsingException(ParsingException):
    def __init__(self, value):
        self.value = value

    def __str__(self):
        return repr(self.value)


class UnknownTimeModification(Exception):
    def __init__(self, value):
        self.value = value

    def __str__(self):
        return repr(self.value)


class LeadsheetTimeSignatureException(Exception):
    def __init__(self, value):
        self.value = value

    def __str__(self):
        return repr(self.value)


class LeadsheetKeySignatureException(Exception):
    def __init__(self, value):
        self.value = value

    def __str__(self):
        return repr(self.value)
