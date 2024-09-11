import datetime
from pathlib import Path
from typing import Annotated, Optional, Union

import typer
from mysoc_validator.models.consts import Chamber, TranscriptType

from .search import ModelHandler, default_model

app = typer.Typer()
yesterday = datetime.date.today() - datetime.timedelta(days=1)

handler = ModelHandler(use_local_model=False)

yesterday = datetime.date.today() - datetime.timedelta(days=1)
last_week = yesterday - datetime.timedelta(days=7)


def parse_date(date: Union[datetime.date, str]) -> datetime.date:
    if isinstance(date, datetime.date):
        return date
    return datetime.date.fromisoformat(date)


DateField = Annotated[datetime.date, typer.Argument(parser=parse_date)]


@app.command()
def search(
    query: str,
    threshold: float = 0.2,
    n: Optional[int] = None,
    start_date: DateField = yesterday,
    end_date: DateField = yesterday,
    chamber: Chamber = Chamber.COMMONS,
    transcript_type: TranscriptType = TranscriptType.DEBATES,
    model_id: str = default_model,
    use_local_model: bool = True,
    override_stored: bool = False,
    dest: Optional[Path] = None,
):
    handler = ModelHandler(
        model_id=model_id,
        use_local_model=use_local_model,
        override_stored=override_stored,
    )

    # if threshold is a float greater than one,  convert to int
    if threshold > 1:
        threshold = int(threshold)

    date_range = ModelHandler.DateRange(start_date=start_date, end_date=end_date)
    results = handler.query(
        query=query,
        threshold=threshold,
        n=n,
        date_range=date_range,
        chamber=chamber,
        transcript_type=transcript_type,
    )

    if dest:
        # if extentsion is json
        if dest.suffix == ".json":
            results.to_path(dest)
        elif dest.suffix == ".csv":
            results.df().to_csv(dest, index=False)
        elif dest.suffix == ".xlsx":
            results.df().to_excel(dest, index=False)
        elif dest.suffix == ".parquet":
            results.df().to_parquet(dest, index=False)
        else:
            raise ValueError(f"Unsupported file type: {dest.suffix}")

    else:
        typer.echo(results.json())


if __name__ == "__main__":
    app()
