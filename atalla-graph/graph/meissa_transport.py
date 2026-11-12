
# takes input as the packer output in ascending group order
# output is 32 512 bit rows, sent for columns [31:0]
# group has four 16 bit lanes: weight0, weight1, position code, padding.
# build with ENABLE_SPARSE_2_4=1
#if needed reference atalla/tb/unit/systolic_array/sysarr_MEISSA_top_tb.sv from line 196

COLUMNS = 32
GROUPS_PER_COLUMN = 8
ROW_BYTES = 64
GROUP_BYTES = 8
TRANSPORT_BYTES = COLUMNS * ROW_BYTES


def to_meissa_transport(value_bytes: bytes, metadata: bytes) -> bytes:

    if not isinstance(value_bytes, bytes) or not isinstance(metadata, bytes):
        raise TypeError("value_bytes and metadata must be bytes")
    if len(value_bytes) != 1024 or len(metadata) != 256:
        raise ValueError("Expected 1024 value bytes and 256 metadata bytes")
    for code in metadata:
        p0, p1 = code & 3, (code >> 2) & 3
        if code & 0xF0 or p0 >= p1:
            raise ValueError("Invalid sorted position pair")

    transport = bytearray(TRANSPORT_BYTES)
    for transfer_row in range(COLUMNS):
        column = COLUMNS - 1 - transfer_row
        for group in range(GROUPS_PER_COLUMN):
            compact_group = GROUPS_PER_COLUMN * column + group
            src = 4 * compact_group
            dst = ROW_BYTES * transfer_row + GROUP_BYTES * group
            transport[dst:dst + 4] = value_bytes[src:src + 4]
            transport[dst + 4] = metadata[compact_group]
            # Three remaining bytes stay zero, including the metadata (padding)
            # halfword's high byte and the fourth 16-bit lane.
    return bytes(transport)
