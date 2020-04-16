import os
import sys
import json
import tempfile
import subprocess
import importlib.util
from typing import Any, Dict, List, Type, Tuple, Union, Iterator
from pathlib import Path
from datetime import datetime
from subprocess import PIPE, Popen

import numpy as np
import pandas as pd
from xdg.BaseDirectory import save_data_path


def import_config():
    """Import module as object.

    src: https://docs.python.org/3/library/importlib.html#importing-a-source-file-directly

    """

    file_path = "buglog/config.py"
    module_name = "config"

    spec = importlib.util.spec_from_file_location(module_name, file_path)
    module = importlib.util.module_from_spec(spec)
    sys.modules[module_name] = module
    spec.loader.exec_module(module)

    return module


# Import Bug class and, thus, all it's descendants
module = import_config()
Bug = module.Bug


def bug_properties(bug_cls: Type[Bug]) -> Dict[str, Any]:
    """Extract properties from type of bug."""
    return bug_cls.schema()["properties"]


def bug_description(bug_cls: Type[Bug]) -> str:
    """Extract description string from type of bug."""
    return bug_cls.schema()["description"]


def str_to_bug(bug_name: str) -> Type[Bug]:
    """Convert string containing bug class name to a class itself."""
    return {bug_cls.__name__: bug_cls for bug_cls in Bug.__subclasses__()}[
        bug_name
    ]


def bug_to_text(bug_cls: Type[Bug]) -> str:
    """Get prompt text for a bug class."""
    schemas = bug_properties(bug_cls).values()
    titles = (schema["title"] for schema in schemas)
    return "".join(f"# {title}\n\n" for title in titles)


def extract_bug_from_text(bug_cls: Type[Bug], text: str) -> Bug:
    """Extract bug from filled in text."""
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
    """Extract all bugs from filled in text."""
    for bug_cls in bug_classes:
        yield extract_bug_from_text(bug_cls, text)


def db_path() -> Path:
    """Get path to current database."""
    time_str = datetime.now().isoformat(" ", "seconds")
    data_dir = save_data_path("buglog")
    return Path(data_dir) / f"{time_str}.json"


def input_bugs(bug_classes: List[Type[Bug]]) -> Iterator[Bug]:
    """Interactively input bugs."""
    try:
        # Create temporary file and write prompts to it
        with tempfile.NamedTemporaryFile(delete=False) as tf:
            for bug_cls in bug_classes:
                text = bug_to_text(bug_cls)
                tf.write(text.encode())
        # Open the file with vim
        subprocess.call([os.environ.get("EDITOR", "vi"), tf.name])
        # Parse the edited file and extract the Bugs
        with open(tf.name, "r") as tf_:
            text = tf_.read()
            bugs = text_to_bugs(bug_classes, text)
            return bugs

    finally:
        os.remove(tf.name)


def fuzzy_pick_bug() -> List[Type[Bug]]:
    """Use fzf to pick Bug types."""
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


def dump_bugs(path: Union[Path, str], bugs: Iterator[Bug]) -> None:
    """Dump Bugs into a JSON file."""
    data = {bug.__class__.__name__: bug.dict() for bug in bugs}

    with open(path, "w") as fout:
        json.dump(data, fout)


def load_data_series() -> Iterator[Tuple[datetime, Bug]]:
    """Yield timestamps along with bugs stored in the database."""
    data_dir = save_data_path("buglog")
    files = sorted(Path(data_dir).glob("*.json"))
    for file in files:
        time = datetime.fromisoformat(file.stem)
        with open(file) as fin:
            for bug_name, bug_fields in json.load(fin).items():
                bug_cls = str_to_bug(bug_name)
                yield time, bug_cls(**bug_fields)


def load_dataframe():
    """Convert database to pandas dataframe."""

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



