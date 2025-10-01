import pytest
from async_utils import map_async, parallelize
import anyio
import time


@pytest.mark.asyncio
async def test_map_async_correctness():
    async def func(x: int) -> int:
        await anyio.sleep(0.1)
        return x * 2

    results = await map_async(func, range(10))
    assert results == [0, 2, 4, 6, 8, 10, 12, 14, 16, 18]


@pytest.mark.asyncio
async def test_map_async_concurrent():
    async def func(x: int) -> int:
        await anyio.sleep(0.1)
        return x * 2

    start = time.time()
    await map_async(func, range(10))
    assert time.time() - start < 0.2


@pytest.mark.asyncio
async def test_map_async_concurrent_limit():
    num_workers = 0

    async def func(x: int) -> int:
        nonlocal num_workers
        num_workers += 1
        assert num_workers <= 2
        await anyio.sleep(0.1)
        num_workers -= 1
        return x * 2

    await map_async(func, range(10), max_concurrent=2)


@pytest.mark.asyncio
async def test_parallelize():
    async def func_one() -> int:
        await anyio.sleep(0.1)
        return 1

    async def func_two() -> int:
        await anyio.sleep(0.1)
        return 2

    start = time.time()
    result_one, result_two = await parallelize(func_one, func_two)
    assert time.time() - start < 0.15
    assert result_one == 1
    assert result_two == 2
