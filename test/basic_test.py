import torch
import importlib
import time

global uvm_mgr
uvm_mgr = None

_uvmlib = "uvm_pytorch"

def get_uvm_manager():
    """ single instance mgr for unified memory tensors """
    global uvm_mgr
    if uvm_mgr is None:
        uvm_mgr = importlib.import_module(_uvmlib)
    return uvm_mgr

size = 20000
mgr = get_uvm_manager()
# get float tensor
start = time.time()
t = mgr.getManagedTensor(size * 4, (size,))
stop = time.time()
assert t is not None, f"Failed to get UM tensor"

print(f"um tensor allocated in {stop-start:.2f} seconds")

t.fill_(0)

print(f"{t=}")

# prefetch
# mgr.cuda_prefetch(t, size*4, 0)

del t

print(f"Complete")
