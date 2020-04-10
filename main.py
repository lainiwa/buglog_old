
from datetime import datetime
from typing import List

from pydantic import (
    BaseModel,
    NegativeFloat,
    NegativeInt,
    PositiveFloat,
    PositiveInt,
    conbytes,
    condecimal,
    confloat,
    conint,
    conlist,
    constr,
    Field,
)
from collections import namedtuple
from collections import defaultdict
from returns.maybe import Maybe, maybe
from returns.result import safe
from returns.io import impure
from returns.pipeline import is_successful
from returns.io import IO
from functools import partial
from xdg.BaseDirectory import save_data_path
from pathlib import Path
from datetime import datetime
import tempfile
import os
import subprocess


class Bug(BaseModel):
    pass

class Soda(Bug):
    liters: confloat(gt=0)

class Mood(Bug):
    mood: conint(gt=0, lt=5)

class Weight(Bug):
    kg: float = Field(..., title="Your weight in kg", gt=0)

class Learned(Bug):
    summary: str = Field(..., title="Description of a thing you learned lately")
    commentary: str




def bug_properties(cls):
    return cls.schema()["properties"]

def bug_to_text(cls):
    schemas = bug_properties(cls).values()
    titles = (schema['title'] for schema in schemas)
    return ''.join(f"# {title}\n\n" for title in titles)

@safe
def extract_bug_from_text(bug_cls, text):
    fields = {}

    for field, schema in bug_properties(bug_cls).items():
        title = schema['title']

        start =  text.find(f"# {title}\n") + len(f"# {title}\n")
        end = text.find(f"# ", start)

        user_input = text[start:end]
        fields[field] = user_input

    return bug_cls(**fields)


def text_to_bugs(bug_classes, text):
    bugs = map(partial(extract_bug_from_text, text=text), bug_classes)
    return (bug.unwrap() for bug in bugs if is_successful(bug))

@impure
def db_path() -> Path:
    time_str = datetime.now().isoformat(' ', 'seconds')
    return Path(save_data_path('buglog')) / f"{time_str}.dump"

def input_bugs(bug_classes):
    try:
        with tempfile.NamedTemporaryFile(delete=False) as tf:
            for bug_cls in bug_classes:
                text = bug_to_text(bug_cls)
                tf.write(text.encode())

        subprocess.call(["vim", "--", tf.name])

        with open(tf.name, 'r') as tf:
            text = tf.read()
            bugs = text_to_bugs(bug_classes, text)
            print(list(bugs))

    finally:
        os.remove(tf.name)



# xs = text_to_bugs(
# """
# sd
# sda
# # Description of a thing you learned lately
# lol
# kekcheburek
# # Commentary
# kek
# # Mood
# 3
# """
# )

# print(list(xs))

# print(db_path())

# input_bugs(bug_classes=Bug.__subclasses__())
input_bugs(bug_classes=[Learned, Mood])
# for x in xs:
#     print(x)
# print(list(xs))


# for propetry in schema["propetrties"]:
    # print(pro`petrty)
# print()



# external_data = {
#     'id': '123',
#     'signup_ts': '2019-06-01 12:22',
#     'friends': [1, 2, '3']
# }
# user = User(**external_data)
# print(user.id)
# print(repr(user.signup_ts))
# print(user.friends)
# print(user.dict())
