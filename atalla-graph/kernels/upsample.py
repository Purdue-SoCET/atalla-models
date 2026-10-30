from pathlib import Path


def upsample_c(**kwargs) -> str:
    """Return the AtallaC kernel string for 2D nearest-neighbor upsample.

    Algorithm:
    - 2x spatial expansion H x W -> 2H x 2W for all channels.
    - Flat 2-level loop structure:
      Outer loop over total_rows (channels * H_in).
      Inner loop over row_words (W_in / 2).
    - Unpacks two 16-bit BF16 elements from a 32-bit GMEM word:
      x0 = val & 0xFFFF, x1 = (val >> 16) & 0xFFFF.
    - Replicates coordinates horizontally:
      dup0 = (x0 << 16) | x0, dup1 = (x1 << 16) | x1.
    - Replicates coordinates vertically:
      Writes dup0 and dup1 into row 2r and row 2r+1.
    - Zero stack spills guaranteed by register allocation across the 2-level loop.
    """
    kernel_path = Path(__file__).parent / "upsample.c"
    return kernel_path.read_text()
