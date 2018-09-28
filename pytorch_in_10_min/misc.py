import time


def tic():
    globals()['tt'] = time.clock()


def toc():
    print(f'Elapsed time: {time.clock() - globals()["tt"]:.3f} seconds')
