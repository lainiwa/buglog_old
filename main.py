import os
import json
import tempfile
import subprocess
from typing import Any, Dict, List, Type, Iterator
from pathlib import Path
from datetime import datetime
from subprocess import PIPE, Popen

from pydantic import Field, BaseModel
from returns.io import IO, impure
from xdg.BaseDirectory import save_data_path


class Bug(BaseModel):
    pass


class Soda(Bug):
    """Drank soda"""

    liters: float = Field(..., title="How much soda you drank", gt=0)
    name: str = Field(..., title="What was it")


class Mood(Bug):
    """Current mood & feel"""

    mood: int = Field(
        ..., title="How do you feel right now? (1=bad ... 5=great)", ge=1, le=5
    )


class Weight(Bug):
    """Weight today"""

    kg: float = Field(..., title="Your weight in kg", ge=0)


class Learned(Bug):
    """Learned stuff"""

    summary: str = Field(
        ..., title="Description of a thing you learned lately"
    )


def bug_properties(bug_cls: Type[Bug]) -> Dict[str, Any]:
    return bug_cls.schema()["properties"]


def bug_description(bug_cls: Type[Bug]) -> str:
    return bug_cls.schema()["description"]


def str_to_bug(bug_name: str) -> Type[Bug]:
    return {bug_cls.__name__: bug_cls for bug_cls in Bug.__subclasses__()}[
        bug_name
    ]


def bug_to_text(bug_cls: Type[Bug]) -> str:
    schemas = bug_properties(bug_cls).values()
    titles = (schema["title"] for schema in schemas)
    return "".join(f"# {title}\n\n" for title in titles)


def extract_bug_from_text(bug_cls: Type[Bug], text: str) -> Bug:
    fields = {}

    for field, schema in bug_properties(bug_cls).items():
        title = schema["title"]

        start = text.find(f"# {title}\n") + len(f"# {title}\n")
        end = text.find(f"# ", start)
        if end == -1:
            end = len(text)

        user_input = text[start:end].strip()
        fields[field] = user_input

    return bug_cls(**fields)


def text_to_bugs(bug_classes: List[Type[Bug]], text: str) -> Iterator[Bug]:
    for bug_cls in bug_classes:
        yield extract_bug_from_text(bug_cls, text)


@impure
def db_path() -> Path:
    time_str = datetime.now().isoformat(" ", "seconds")
    data_dir = save_data_path("buglog")
    return Path(data_dir) / f"{time_str}.dump"


@impure
def input_bugs(bug_classes: List[Type[Bug]]) -> Iterator[Bug]:
    try:
        with tempfile.NamedTemporaryFile(delete=False) as tf:
            for bug_cls in bug_classes:
                text = bug_to_text(bug_cls)
                tf.write(text.encode())

        subprocess.call([os.environ.get("EDITOR", "vi"), "--", tf.name])

        with open(tf.name, "r") as tf1:
            text = tf1.read()
            bugs = text_to_bugs(bug_classes, text)
            return bugs

    finally:
        os.remove(tf.name)


@impure
def fuzzy_pick_bug() -> List[Type[Bug]]:
    fzf_input = "\n".join(
        f"{cls.__name__} {bug_description(cls)}"
        for cls in Bug.__subclasses__()
    )

    pipe = Popen(
        ["fzf", "--multi", "--with-nth", "2.."], stdout=PIPE, stdin=PIPE
    )
    stdout = pipe.communicate(input=fzf_input.encode())[0]
    return [
        str_to_bug(line.split(maxsplit=1)[0])
        for line in stdout.decode().splitlines()
    ]


@impure
def dump_bugs(bugs: IO[Iterator[Bug]]) -> None:
    data = [{bug.__class__.__name__: bug.dict()} for bug in bugs]

    time_str = datetime.now().isoformat(" ", "seconds")
    data_dir = save_data_path("buglog")
    path = Path(data_dir) / f"{time_str}.dump"

    with open(path, "w") as fout:
        json.dump(data, fout)


bug_classes: IO[List[Type[Bug]]] = fuzzy_pick_bug()
lazy_bugs: IO[Iterator[Bug]] = bug_classes.bind(input_bugs)
bugs: IO[List[Bug]] = lazy_bugs.map(list)
bugs.bind(dump_bugs)
