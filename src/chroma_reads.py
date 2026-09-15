"""Bound Chroma reads to avoid backend SQL parameter limits on large libraries."""


def read_collection(collection, include, where=None, page_size=500):
    """Collect aligned rows in bounded pages, retaining the optional source filter.

    This is not a database snapshot: callers must retain their revision checks or
    writer lock. Duplicate IDs across pages fail rather than returning mixed data.
    """
    if not 1 <= page_size <= 1000:
        raise ValueError("page_size must be in [1, 1000]")
    result = {"ids": [], **{key: [] for key in include}}
    seen = set()
    offset = 0
    while True:
        page = collection.get(include=include, where=where, limit=page_size, offset=offset)
        ids = page.get('ids', [])
        if not ids:
            break
        if len(ids) != len(set(ids)) or seen.intersection(ids):
            raise RuntimeError("Collection changed during paginated read; retry")
        for key in include:
            values = page.get(key)
            if values is None or len(values) != len(ids):
                raise RuntimeError(f"Incomplete {key} in collection read; refusing partial data")
            result[key].extend(values)
        result['ids'].extend(ids)
        seen.update(ids)
        offset += len(ids)
    return result
