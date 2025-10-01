import anyio
import typing

_T = typing.TypeVar("_T")
_V = typing.TypeVar("_V")


async def map_async(
    func: typing.Callable[[_T], typing.Awaitable[_V]],
    items: typing.Iterable[_T],
    *,
    max_concurrent: int = 10,
) -> list[_V]:
    semaphore = anyio.Semaphore(max_concurrent)
    results = dict[int, _V]()

    async def worker(item: _T, index: int):
        async with semaphore:
            result = await func(item)
            results[index] = result

    async with anyio.create_task_group() as tg:
        for index, item in enumerate(items):
            tg.start_soon(worker, item, index)

    return [results[i] for i, _ in enumerate(items)]


async def parallelize(
    func_one: typing.Callable[[], typing.Awaitable[_T]],
    func_two: typing.Callable[[], typing.Awaitable[_V]],
) -> tuple[_T, _V]:
    results = dict[typing.Literal[0, 1], _V | _T]()

    async def worker(index: typing.Literal[0, 1]):
        result = await func_one() if index == 0 else await func_two()
        results[index] = result

    async with anyio.create_task_group() as tg:
        tg.start_soon(worker, 0)
        tg.start_soon(worker, 1)

    return results[0], results[1]
