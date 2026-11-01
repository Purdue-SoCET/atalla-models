/* Runtime-configured, same-dtype data movement. All geometry is in config.
 * Config words at byte 60: source, destination, rank, elements, element bytes,
 * snapshot, then rank-sized arrays: extents, source strides, destination strides.
 * Stage 0 snapshots every logical source element before stage 1 writes output.
 * Aligned word read/modify/write preserves neighboring packed values exactly.
 */
int main() {
    int *cfg = (int *)60;
    int source = cfg[0];
    int destination = cfg[1];
    int rank = cfg[2];
    int elements = cfg[3];
    int element_bytes = cfg[4];
    int snapshot = cfg[5];
    int *extents = cfg + 6;
    int *source_strides = extents + rank;
    int *destination_strides = source_strides + rank;

    int stage = 0;
    while (stage < 2) {
        int element = 0;
        while (element < elements) {
            int linear = element;
            int source_address = source;
            int destination_address = destination;
            int axis = rank - 1;
            while (axis >= 0) {
                int coordinate = linear % extents[axis];
                linear = linear / extents[axis];
                source_address = source_address + coordinate * source_strides[axis];
                destination_address = destination_address + coordinate * destination_strides[axis];
                axis = axis - 1;
            }
            if (stage == 0) {
                destination_address = snapshot + element * element_bytes;
            } else {
                source_address = snapshot + element * element_bytes;
            }
            int byte = 0;
            while (byte < element_bytes) {
                int read_address = source_address + byte;
                int write_address = destination_address + byte;
                int read_shift = (read_address & 3) * 8;
                int write_shift = (write_address & 3) * 8;
                unsigned int word = *(volatile unsigned int *)(read_address & -4);
                unsigned int value = (word >> read_shift) & 255;
                unsigned int mask = 255;
                mask = mask << write_shift;
                unsigned int old = *(volatile unsigned int *)(write_address & -4);
                unsigned int updated = (old & ~mask) | (value << write_shift);
                *(volatile unsigned int *)(write_address & -4) = updated;
                byte = byte + 1;
            }
            element = element + 1;
        }
        stage = stage + 1;
    }
    atalla_halt();
    return 0;
}
