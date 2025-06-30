import asyncio
import pytest
from src.optimization.parallel_execution import ParallelExecutor, ParallelTask


async def _async_add(x, y):
    await asyncio.sleep(0.01)
    return x + y


def _sync_mul(x, y):
    return x * y


@pytest.mark.asyncio
async def test_simple_task_execution():
    exec = ParallelExecutor(max_workers=2)

    task = ParallelTask(
        task_id="t1",
        task_type="unit",
        function=_sync_mul,
        args=(2, 3),
        kwargs={},
    )

    await exec.submit_task(task)
    await asyncio.sleep(0.05)

    stats = exec.get_stats()
    assert stats["completed_tasks"] >= 1
    exec.shutdown()


@pytest.mark.asyncio
async def test_cache_hit():
    exec = ParallelExecutor(max_workers=2)

    t1 = ParallelTask("c1", "unit", _sync_mul, (4, 5), {})
    t2 = ParallelTask("c2", "unit", _sync_mul, (4, 5), {})  # Mesma operação → cache

    await exec.submit_task(t1)
    await exec.submit_task(t2)
    await asyncio.sleep(0.05)

    stats = exec.get_stats()
    assert stats["cache_hits"] >= 1
    exec.shutdown()


@pytest.mark.asyncio
async def test_dependency_execution():
    exec = ParallelExecutor(max_workers=2)

    t1 = ParallelTask("d1", "unit", _sync_mul, (1, 2), {})

    async def _after():
        return 42

    t2 = ParallelTask("d2", "unit", _after, (), {}, dependencies={"d1"})

    await exec.submit_task(t1)
    await exec.submit_task(t2)

    await asyncio.sleep(0.1)
    stats = exec.get_stats()
    assert stats["completed_tasks"] >= 2
    exec.shutdown() 