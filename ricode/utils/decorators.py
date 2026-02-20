import attr

from ricode.ml.training_basics import conf_to_json


def attrs_to_json(cls):
    if not attr.has(cls):
        raise ValueError(cls)

    if not hasattr(cls, "to_json"):

        def to_json(self):
            return conf_to_json(self)

        setattr(cls, "to_json", to_json)
    return cls
