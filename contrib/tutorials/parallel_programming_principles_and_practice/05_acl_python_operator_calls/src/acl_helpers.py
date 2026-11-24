"""管理 pyACL 示例的设备资源；算子调用和数值校验保留在 Notebook 中。"""
import acl

ACL_FLOAT = 0
ACL_FLOAT16 = 1
ACL_INT64 = 9
ACL_FORMAT_ND = 2
ACL_COMPUTE_HIGH_PRECISION = 0
MEMCPY_H2D = 1
MEMCPY_D2H = 2
MALLOC_FLAG = 2


def check(ret, call):
    if ret != 0:
        raise RuntimeError(f"{call} 失败，ret={ret}")


class AclSession:
    """一次计算使用的 Context、Stream、Event 和内存，结束时统一释放。"""

    def __init__(self, model_dir, device_id):
        self.model_dir = model_dir
        self.device_id = device_id
        self.context = self.stream = self.start_event = self.end_event = None
        self.acl_inited = self.device_set = False
        self.ptrs = []
        self.tensor_objects = []
        self.attr = None

    def __enter__(self):
        try:
            check(acl.init(), "acl.init")
            self.acl_inited = True
            check(acl.op.set_model_dir(str(self.model_dir)), "acl.op.set_model_dir")
            check(acl.rt.set_device(self.device_id), "acl.rt.set_device")
            self.device_set = True
            self.context, ret = acl.rt.create_context(self.device_id)
            check(ret, "acl.rt.create_context")
            self.stream, ret = acl.rt.create_stream()
            check(ret, "acl.rt.create_stream")
            self.start_event, ret = acl.rt.create_event()
            check(ret, "acl.rt.create_event start")
            self.end_event, ret = acl.rt.create_event()
            check(ret, "acl.rt.create_event end")
            self.current_device, ret = acl.rt.get_device()
            check(ret, "acl.rt.get_device")
            return self
        except BaseException:
            self.close()
            raise

    def to_device(self, array):
        ptr, ret = acl.rt.malloc(array.nbytes, MALLOC_FLAG)
        check(ret, "acl.rt.malloc")
        self.ptrs.append(ptr)
        host_bytes = array.tobytes()
        check(acl.rt.memcpy(
            ptr, array.nbytes, acl.util.bytes_to_ptr(host_bytes), array.nbytes, MEMCPY_H2D
        ), "acl.rt.memcpy H2D")
        return ptr

    def create_tensor(self, ptr, array, dtype):
        desc = acl.create_tensor_desc(dtype, list(array.shape), ACL_FORMAT_ND)
        self.tensor_objects.append((desc, None))
        buffer = acl.create_data_buffer(ptr, array.nbytes)
        self.tensor_objects[-1] = (desc, buffer)
        if desc is None or buffer is None:
            raise RuntimeError("创建 TensorDesc/DataBuffer 失败")
        return desc, buffer

    def input_tensor(self, array, dtype):
        ptr = self.to_device(array)
        return self.create_tensor(ptr, array, dtype)

    def output_tensor(self, array, dtype):
        ptr, ret = acl.rt.malloc(array.nbytes, MALLOC_FLAG)
        check(ret, "acl.rt.malloc output")
        self.ptrs.append(ptr)
        desc, buffer = self.create_tensor(ptr, array, dtype)
        return ptr, desc, buffer

    def create_attr(self):
        self.attr = acl.op.create_attr()
        if self.attr is None:
            raise RuntimeError("acl.op.create_attr 失败")
        return self.attr

    def close(self):
        for event in (self.end_event, self.start_event):
            if event is not None:
                acl.rt.destroy_event(event)
        if self.attr is not None:
            acl.op.destroy_attr(self.attr)
        for desc, buffer in reversed(self.tensor_objects):
            if buffer is not None:
                acl.destroy_data_buffer(buffer)
            if desc is not None:
                acl.destroy_tensor_desc(desc)
        for ptr in reversed(self.ptrs):
            acl.rt.free(ptr)
        if self.stream is not None:
            acl.rt.destroy_stream(self.stream)
        if self.context is not None:
            acl.rt.destroy_context(self.context)
        if self.device_set:
            acl.rt.reset_device(self.device_id)
        if self.acl_inited:
            acl.finalize()

    def __exit__(self, exc_type, exc, traceback):
        self.close()
