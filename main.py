import os
import json
import tempfile
import subprocess
from typing import Any, Dict, List, Type, Tuple, Union, Iterator
from pathlib import Path
from datetime import datetime
from subprocess import PIPE, Popen

import numpy as np
import typer
import pandas as pd
from pydantic import Field, BaseModel
from returns.io import IO, impure
from xdg.BaseDirectory import save_data_path
from returns.converters import squash_io


class Bug(BaseModel):
    pass


class Soda(Bug):
    """Drank soda"""

    liters: float = Field(
        ..., title="How much soda you drank (in liters)", gt=0
    )
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
    return Path(data_dir) / f"{time_str}.json"


@impure
def input_bugs(bug_classes: List[Type[Bug]]) -> Iterator[Bug]:
    try:
        # Create temporary file and write prompts to it
        with tempfile.NamedTemporaryFile(delete=False) as tf:
            for bug_cls in bug_classes:
                text = bug_to_text(bug_cls)
                tf.write(text.encode())
        # Open the file with vim
        subprocess.call([os.environ.get("EDITOR", "vi"), tf.name])
        # Parse the edited file and extract the Bugs
        with open(tf.name, "r") as tf1:
            text = tf1.read()
            bugs = text_to_bugs(bug_classes, text)
            return bugs

    finally:
        os.remove(tf.name)


@impure
def fuzzy_pick_bug() -> List[Type[Bug]]:
    """Use fzf to pick Bugs.

    Returns:
        User-picked Bug types.

    """
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
def dump_bugs(path: Union[Path, str], bugs: Iterator[Bug]) -> None:
    """Dump Bugs into a JSON file."""
    data = {bug.__class__.__name__: bug.dict() for bug in bugs}

    with open(path, "w") as fout:
        json.dump(data, fout)


def apply_multiple(fun, *args):
    """Call function from pure arguments with IO arguments."""
    new_fun = lambda tpl: fun(*tpl)
    return squash_io(*args).bind(new_fun)


def load_data_series() -> Iterator[Tuple[datetime, Bug]]:
    data_dir = save_data_path("buglog")
    files = sorted(Path(data_dir).glob("*.json"))
    for file in files:
        time = datetime.fromisoformat(file.stem)
        with open(file) as fin:
            for bug_name, bug_fields in json.load(fin).items():
                bug_cls = str_to_bug(bug_name)
                yield time, bug_cls(**bug_fields)


def load_dataframe():
    def _indexes():
        yield ("timestamp",)
        for bug_cls in Bug.__subclasses__():
            for prop in bug_properties(bug_cls).keys():
                yield (bug_cls.__name__, prop)

    # Create empty dataframe
    columns = pd.MultiIndex.from_tuples(_indexes())
    df = pd.DataFrame(columns=columns)
    # Index by time
    df = df.set_index(("timestamp", np.nan))
    df.index.name = "timestamp"
    #  For each record
    for timestamp, bug in load_data_series():
        # for each bug in record
        for bug_field, field_value in bug.dict().items():
            df.loc[
                timestamp, (bug.__class__.__name__, bug_field)
            ] = field_value

    return df

    # print(df.dtypes)
    # df = df.astype({
    #     ('timestamp', np.nan): 'datetime64[ns]',
    #     ('Soda', 'liters'):'int32'
    #     })
    # print(df.dtypes)

    # df.index = df.index.astype('datetime64')


app = typer.Typer()


@app.command()
def new():
    path = db_path()

    bug_classes: IO[List[Type[Bug]]] = fuzzy_pick_bug()
    lazy_bugs: IO[Iterator[Bug]] = bug_classes.bind(input_bugs)
    bugs: IO[List[Bug]] = lazy_bugs.map(list)

    # bugs = IO([Soda(liters=2, name="coke"), Mood(mood=3)])

    apply_multiple(dump_bugs, path, bugs)


@app.command()
def pr():
    df = load_dataframe()
    print(df)


if __name__ == "__main__":
    app()
