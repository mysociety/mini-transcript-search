from __future__ import annotations

import datetime
from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path
from typing import ClassVar, Iterator, Literal, NamedTuple, Optional, Type, Union

import numpy as np
import pandas as pd
from mysoc_validator.models.consts import Chamber, TranscriptType
from mysoc_validator.models.transcripts import (
    MajorHeading,
    MinorHeading,
    OralHeading,
    Speech,
    Transcript,
)
from mysoc_validator.utils.parlparse.downloader import get_latest_for_date
from pydantic import BaseModel, computed_field
from tqdm import tqdm

from .inference import Inference

default_model = "BAAI/bge-small-en-v1.5"


def twfy_alias(chamber: Chamber) -> str:
    # Alias for internal debate storage

    if chamber == Chamber.COMMONS:
        return "debates"
    elif chamber == Chamber.LORDS:
        return "lords"
    elif chamber == Chamber.SCOTLAND:
        return "sp"
    elif chamber == Chamber.SENEDD:
        return "senedd"
    elif chamber == Chamber.NORTHERN_IRELAND:
        return "ni"
    else:
        raise ValueError(f"Invalid chamber {chamber}")


class DateRange(NamedTuple):
    start_date: datetime.date
    end_date: datetime.date


@dataclass
class Criteria:
    search_queries: list[str]
    score_type: Literal["nearest", "first", "agg", "avg"] = "first"
    search_vectors: Optional[list[np.ndarray]] = None

    def __str__(self):
        return self.search_queries[0]

    def calculate_search_vectors(self, inference: Inference):
        if self.score_type == "first" and len(self.search_queries) > 1:
            raise ValueError("Cannot use 'first' score type with multiple queries")

        self.search_vectors = inference.query(self.search_queries)

        if self.score_type == "avg":
            self.search_vectors = [np.mean(self.search_vectors, axis=0)]

    def get_single_distance(self, vector: np.ndarray) -> float:
        if not self.search_vectors:
            raise ValueError("No search vectors calculated")

        matches = []
        for search_vector in self.search_vectors:
            matches.append(cosine_distance(search_vector, vector))
        # for first, this is easy
        if self.score_type in ["first", "avg"]:
            return matches[0]

        # for nearest, we want to return the closest distance in any of the search queries
        elif self.score_type == "nearest":
            return min(matches)

        elif self.score_type == "agg":
            return np.sqrt(sum([x**2 for x in matches]))
        else:
            raise ValueError(f"Invalid score type {self.score_type}")


class Match(BaseModel):
    distance: float
    speech_id: str
    speaker_name: Optional[str]
    person_id: Optional[str]
    chamber: Chamber
    matched_text: str

    @computed_field
    @property
    def debate_url(self) -> str:
        twfy_slug = twfy_alias(self.chamber)
        debate_id, paragraph_id = (
            self.speech_id.split("#")
            if "#" in self.speech_id
            else (self.speech_id, None)
        )
        url_id = debate_id.split("/")[-1]
        return f"https://www.theyworkforyou.com/{twfy_slug}/?id={url_id}"

    @computed_field
    @property
    def member_url(self) -> str:
        person_id = self.person_id.split("/")[-1] if self.person_id else None
        if person_id is None:
            return ""
        return f"https://www.theyworkforyou.com/mp/{person_id}"


class SearchResult(BaseModel):
    search_query: str
    date_range: DateRange
    threshold: Union[float, int]
    chamber: Chamber
    transcript_type: TranscriptType
    matches: list[Match]

    def df(self) -> pd.DataFrame:
        rows = []
        for match in self.matches:
            rows.append(
                {
                    "search_query": self.search_query,
                    "date_range_start": self.date_range.start_date,
                    "date_range_end": self.date_range.end_date,
                    "distance": match.distance,
                    "matched_text": match.matched_text,
                    "speaker_name": match.speaker_name,
                    "person_id": match.person_id,
                    "chamber": self.chamber,
                    "transcript_type": self.transcript_type,
                    "speech_id": match.speech_id,
                    "debate_url": match.debate_url,
                }
            )
        return pd.DataFrame(rows)

    def to_json(self):
        return self.model_dump_json(indent=2)

    def to_path(self, path: Path):
        path.write_text(self.to_json())


def cosine_distance(vector1: np.ndarray, vector2: np.ndarray) -> float:
    """
    Calcualte cosine distance
    Between 0 and 1 - closer to 0 is more similar.
    """
    cosine_similarity = np.dot(vector1, vector2) / (
        np.linalg.norm(vector1) * np.linalg.norm(vector2)
    )

    return 1 - cosine_similarity


@lru_cache
def get_id_lookup(
    date: datetime.date, chamber: Chamber, transcript_type: TranscriptType
) -> dict[str, Union[Speech, MinorHeading, MajorHeading, OralHeading]]:
    t = Transcript.from_parlparse(date, chamber, transcript_type)
    id_lookup = {
        x.id: x
        for x in t.iter_has_text()
        if isinstance(x, (Speech, MinorHeading, MajorHeading, OralHeading))
    }
    return id_lookup


def iter_headings_and_paragraphs(transcript: Transcript) -> Iterator[tuple[str, str]]:
    """
    Iter through paragraphs rather than contents of the transcript
    """
    for speech in transcript.iter_has_text():
        if isinstance(speech, Speech):
            for paragraph in speech.items:
                s_id = speech.id
                if paragraph.pid:
                    s_id += f"#{paragraph.pid}"
                yield (
                    s_id,
                    str(paragraph).strip(),
                )
                # split on sentences and yield that too
                sentences = str(paragraph).replace("hon.", "hon").split(".")
                for i, sentence in enumerate(sentences):
                    if sentence.strip():
                        yield (
                            f"{s_id}.{i}",
                            sentence.strip(),
                        )

        elif isinstance(speech, (MinorHeading, MajorHeading, OralHeading)):
            yield speech.id, str(speech)


def speech_from_id(
    date: datetime.date, id: str, chamber: Chamber, transcript_type: TranscriptType
):
    speech_id, para_id = id.split("#") if "#" in id else (id, None)
    id_lookup = get_id_lookup(date, chamber, transcript_type)
    final_part = speech_id.split("/")[-1]
    prior_parts = "/".join(speech_id.split("/")[:-1])
    final_part_dots = final_part.split(".")

    # date, major, minor (and then the sentence we want to remove)
    if len(final_part_dots) > 3:
        speech_id = prior_parts + "/" + ".".join(final_part_dots[:3])
    result = id_lookup.get(speech_id)
    if result is None:
        raise ValueError(f"Speech with id {speech_id} not found")
    return result


@dataclass
class ModelHandler:
    DateRange: ClassVar[Type[DateRange]] = DateRange
    Chamber: ClassVar[Type[Chamber]] = Chamber
    TranscriptType: ClassVar[Type[TranscriptType]] = TranscriptType
    model_id: str = default_model
    hf_token: Optional[str] = None
    use_local_model: bool = False
    override_stored: bool = False
    _model: Optional[Inference] = None
    silent: bool = False
    storage_path: Path = Path("data")

    def get_inference(self) -> Inference:
        if self._model is None:
            self._model = Inference(
                self.model_id, hf_token=self.hf_token, local=self.use_local_model
            )
        return self._model

    def query(
        self,
        query: Union[Criteria, str],
        *,
        threshold: float = 1.0,
        n: Optional[int] = None,
        date_range: DateRange,
        chamber: Chamber = Chamber.COMMONS,
        transcript_type: TranscriptType = TranscriptType.DEBATES,
    ):
        """
        Run a query against the transcripts in the given date range.

        Threshold can either be an int and will filter the cosine similarity.
        or a int that will take the top n results.

        """

        if isinstance(query, str):
            query = Criteria(search_queries=[query], score_type="first")

        query.calculate_search_vectors(self.get_inference())

        df = self.get_multi_day_vector(date_range, chamber, transcript_type)

        df["cosine_similarity"] = df["embedding"].apply(query.get_single_distance)

        df = df.sort_values(by="cosine_similarity", ascending=True)

        df = df[df["cosine_similarity"] < threshold]
        if n:
            df = df.head(n)

        records = df.to_dict(orient="records")  # type: ignore

        matches = []
        for r in records:
            date_obj = datetime.date.fromisoformat(r["date"])
            rel_speech = speech_from_id(date_obj, r["id"], chamber, transcript_type)
            if isinstance(rel_speech, Speech):
                speaker_name = rel_speech.speakername
                person_id = rel_speech.person_id
            else:
                speaker_name = None
                person_id = None
            matches.append(
                Match(
                    chamber=chamber,
                    distance=r["cosine_similarity"],
                    speech_id=r["id"],
                    speaker_name=speaker_name,
                    person_id=person_id,
                    matched_text=r["text"],
                )
            )

        return SearchResult(
            search_query=str(query),
            threshold=threshold,
            date_range=date_range,
            chamber=chamber,
            transcript_type=transcript_type,
            matches=matches,
        )

    def get_multi_day_vector(
        self,
        date_range: DateRange,
        chamber: Chamber = Chamber.COMMONS,
        transcript_type: TranscriptType = TranscriptType.DEBATES,
    ):
        # get all week days between start and end date
        dates = []
        current_date = date_range.start_date
        while current_date <= date_range.end_date:
            if current_date.weekday() < 5:
                dates.append(current_date)
            current_date += datetime.timedelta(days=1)

        dfs = []
        for date in tqdm(dates, disable=self.silent):
            try:
                df = self.get_transcript_parquet(
                    date,
                    chamber=chamber,
                    transcript_type=transcript_type,
                )
            except FileNotFoundError:
                continue
            df["date"] = date.isoformat()
            dfs.append(df)
        if len(dfs) == 0:
            raise FileNotFoundError("No transcripts found for given date range")
        return pd.concat(dfs)

    def get_transcript_parquet(
        self,
        date: datetime.date,
        chamber: Chamber = Chamber.COMMONS,
        transcript_type: TranscriptType = TranscriptType.DEBATES,
    ) -> pd.DataFrame:
        transcript_path = get_latest_for_date(
            date,
            transcript_type=transcript_type,
            chamber=chamber,
            download_path=self.storage_path,
        )
        parquet_path = transcript_path.with_suffix(".parquet")
        if not parquet_path.exists() or self.override_stored:
            transcript = Transcript.from_xml_path(transcript_path)
            data = dict(iter_headings_and_paragraphs(transcript))
            df = self.get_inference().query_id_and_text(data)
            df.to_parquet(parquet_path)
        else:
            df = pd.read_parquet(parquet_path)
        return df
