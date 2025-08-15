from concurrent.futures import Future, ThreadPoolExecutor, as_completed
from typing import Callable, Iterable, TypeVar, Concatenate, ParamSpec
from tqdm import tqdm
import bs4

_T = TypeVar("_T")
_V_co = TypeVar("_V_co", covariant=True)
_P = ParamSpec("_P")


def map_threaded(
    fn: Callable[Concatenate[_T, _P], _V_co],
    it: Iterable[_T],
    /,
    max_concurrency: int,
    show_progress: bool | str = False,
    *args: _P.args,
    **kwargs: _P.kwargs,
) -> list[_V_co]:
    """Map a function over an iterable using threads, optionally limiting concurrency.

    This is the threaded analogue of an async map. It preserves input order in the
    returned results and can display a progress bar via ``tqdm``.

    Args:
        fn: A callable that will be applied to each item. It receives the item as
            the first argument followed by any additional ``*args`` and ``**kwargs``.
        it: An iterable of input items.
        max_concurrency: Maximum number of worker threads. If ``None``, uses the
            default for ``ThreadPoolExecutor``.
        show_progress: If ``True`` shows a progress bar with default description.
            If a string is provided, it is used as the progress bar description.
        *args: Positional arguments forwarded to ``fn``.
        **kwargs: Keyword arguments forwarded to ``fn``.

    Returns:
        list[_V_co]: Results in the same order as the input iterable.
    """

    results: dict[int, _V_co] = {}

    pbar = None
    if show_progress:
        desc = "Processing" if show_progress is True else str(show_progress)
        total: int | None = len(list(it))
        pbar = tqdm(total=total, desc=desc)

    # Use ThreadPoolExecutor for concurrent mapping
    # If max_concurrency is None, let the executor choose a sensible default
    with ThreadPoolExecutor(max_workers=max_concurrency) as executor:
        future_to_index: dict[Future[_V_co], int] = {}
        for idx, value in enumerate(it):
            future = executor.submit(fn, value, *args, **kwargs)
            future_to_index[future] = idx

        for future in as_completed(future_to_index):
            idx = future_to_index[future]
            results[idx] = future.result()
            if pbar is not None:
                pbar.update(1)

    if pbar is not None:
        pbar.close()

    return [results[i] for i in range(len(results))]


def extract_xml_tag(content: str, tag: str):
    soup = bs4.BeautifulSoup(content, features="html.parser")
    match = soup.find(tag)
    return match.get_text() if match else None
