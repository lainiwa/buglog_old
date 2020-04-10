
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
from subprocess import Popen, PIPE, STDOUT


class Bug(BaseModel):
    pass

class Soda(Bug):
    """Drank soda"""
    liters: confloat(gt=0)
    tea: int

class Mood(Bug):
    """Mood: how was your day"""
    mood: conint(gt=0, lt=5)

class Weight(Bug):
    """Weight today"""
    kg: float = Field(..., title="Your weight in kg", gt=0)

class Learned(Bug):
    """Learned stuff"""
    summary: str = Field(..., title="Description of a thing you learned lately")
    commentary: str




def bug_properties(bug_cls):
    return bug_cls.schema()["properties"]

def bug_description(bug_cls):
    return bug_cls.schema()["description"]

def str_to_bug(bug_name):
    return {cls.__name__: cls for cls in Bug.__subclasses__()}[bug_name]

def bug_to_text(bug_cls):
    schemas = bug_properties(bug_cls).values()
    titles = (schema['title'] for schema in schemas)
    return ''.join(f"# {title}\n\n" for title in titles)

def extract_bug_from_text(bug_cls, text):
    fields = {}

    for field, schema in bug_properties(bug_cls).items():
        title = schema['title']

        start =  text.find(f"# {title}\n") + len(f"# {title}\n")
        end = text.find(f"# ", start)
        if end == -1:
            end = len(text)

        user_input = text[start:end].strip()
        fields[field] = user_input

    return bug_cls(**fields)


def text_to_bugs(bug_classes, text):
    for bug_cls in bug_classes:
        yield extract_bug_from_text(bug_cls, text)

@impure
def db_path() -> Path:
    time_str = datetime.now().isoformat(' ', 'seconds')
    return Path(save_data_path('buglog')) / f"{time_str}.dump"

@impure
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
            return bugs

    finally:
        os.remove(tf.name)

@impure
def fuzzy_pick_bug():
    fzf_input = '\n'.join(
        f"{cls.__name__} {bug_description(cls)}"
        for cls in Bug.__subclasses__())

    pipe = Popen(['fzf', '--multi', '--with-nth', '2..'], stdout=PIPE, stdin=PIPE)
    stdout = pipe.communicate(input=fzf_input.encode())[0]
    return [str_to_bug(line.split(maxsplit=1)[0]) for line in stdout.decode().splitlines()]



bug_classes = fuzzy_pick_bug()
bugs = bug_classes.bind(input_bugs).map(list)
print(bugs)
# input_bugs(bug_classes=bug_classes)


# subprocess.call(["fzf"])

# proc = subprocess.run(["fzf"], universal_newlines=True, stdin=subprocess.PIPE, stdout=subprocess.PIPE)
# proc.stdin.write("hello\nhello world\nhella")

# print(proc.stdout)

# proc.stdin.close()


# pipe = Popen(['fzf'])
# grep_stdout = pipe.communicate(input=b'one\ntwo\nthree\nfour\nfive\nsix\n')[0]
# print(grep_stdout.decode())

# print(fzf_input)

# for bug_cls in Bug.__subclasses__():
#     description = bug_cls.schema()["description"]
#     print(description)


# xs = extract_bug_from_text(Mood,
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

# print(xs)
# exit(11)

# print(db_path())

# input_bugs(bug_classes=Bug.__subclasses__())
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
