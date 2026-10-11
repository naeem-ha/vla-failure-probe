"""Clear the executable-stack flag (PT_GNU_STACK PF_X) on ELF shared objects.

Old torch wheels (cu113) request an executable stack, which glibc >= 2.41
refuses ("cannot enable executable stack"). Equivalent to `patchelf --clear-execstack`.
Usage: python clear_execstack.py <lib.so> [...]
"""
import struct
import sys

PT_GNU_STACK, PF_X = 0x6474E551, 0x1

for path in sys.argv[1:]:
    with open(path, "r+b") as f:
        hdr = f.read(64)
        if hdr[:4] != b"\x7fELF" or hdr[4] != 2:  # 64-bit only
            continue
        phoff, = struct.unpack_from("<Q", hdr, 0x20)
        phentsize, phnum = struct.unpack_from("<HH", hdr, 0x36)
        for i in range(phnum):
            off = phoff + i * phentsize
            f.seek(off)
            p_type, p_flags = struct.unpack("<II", f.read(8))
            if p_type == PT_GNU_STACK and p_flags & PF_X:
                f.seek(off + 4)
                f.write(struct.pack("<I", p_flags & ~PF_X))
                print("cleared:", path)
