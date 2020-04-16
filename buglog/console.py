import typer
from buglog.util import db_path, fuzzy_pick_bug, input_bugs, dump_bugs, load_dataframe

app = typer.Typer()

@app.command()
def new():
    path = db_path()

    bug_classes = fuzzy_pick_bug()
    if not bug_classes:
        return
    bugs = input_bugs(bug_classes)
    dump_bugs(path, bugs)


@app.command()
def pr():
    df = load_dataframe()
    print(df)


if __name__ == "__main__":
    app()
