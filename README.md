# mini-transcript-search

This is an experiment in low dependency vector search.

For just checking a few days - we don't need an index or a big database. We can just calculate cosine similarity directly.

See infer.ipynb for usage as a module.

```bash
python -m mini_transcript_search "register of members financial interests" --threshold 0.4 --n 5 --dest test.json
```

uvx:

```bash
uvx --from git+https://github.com/mysociety/mini-transcript-search transcript-search "register of members financial interests" --threshold 0.4 --n 5
```