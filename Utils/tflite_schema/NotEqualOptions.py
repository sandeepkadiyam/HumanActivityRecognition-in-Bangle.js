# automatically generated by the FlatBuffers compiler, do not modify

# namespace: tflite

import flatbuffers

class NotEqualOptions(object):
    __slots__ = ['_tab']

    @classmethod
    def GetRootAsNotEqualOptions(cls, buf, offset):
        n = flatbuffers.encode.Get(flatbuffers.packer.uoffset, buf, offset)
        x = NotEqualOptions()
        x.Init(buf, n + offset)
        return x

    @classmethod
    def NotEqualOptionsBufferHasIdentifier(cls, buf, offset, size_prefixed=False):
        return flatbuffers.util.BufferHasIdentifier(buf, offset, b"\x54\x46\x4C\x33", size_prefixed=size_prefixed)

    # NotEqualOptions
    def Init(self, buf, pos):
        self._tab = flatbuffers.table.Table(buf, pos)

def NotEqualOptionsStart(builder): builder.StartObject(0)
def NotEqualOptionsEnd(builder): return builder.EndObject()
