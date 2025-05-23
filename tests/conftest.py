import asyncio

def pytest_pyfunc_call(pyfuncitem):
    if "asyncio" in pyfuncitem.keywords:
        asyncio.run(pyfuncitem.obj(**pyfuncitem.funcargs))
        return True
