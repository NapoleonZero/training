import numpy as np
import pandas as pd
import torch

def string_to_matrix(bitboard):
    return np.array([b for b in bitboard], dtype=np.uint8).reshape(8,8).copy()

def string_to_array(bitboard):
    return np.array([b for b in bitboard], dtype=np.uint8)

def uint_to_bits(x, bits = 64):
    return np.unpackbits(np.array([x], dtype='>u8').view(np.uint8))

# TODO: move these utility functions to a separate file
def read_bitboards(csv):
  """ csv: comma separated set of bitboards.
           Each element of the set is a string of 64 binary values.

      returns: np.array of shape 12x8x8
  """
  bitboards = csv.split(',')
  return np.array([string_to_matrix(b) for b in bitboards])

def read_positions(file: str):
    """ Read positions from `file` and return the following information:
        - descriptors (either fen of description string) list
        - positions (bitboards) tensor
        - auxiliary inputs (side, ep square, castling) tensor
        - scores tensor
    """
    dtypes: dict = {
            0: 'string',                # fen position string
            1: 'string', 2: 'string',   # bitboards
            3: 'string', 4: 'string',   # bitboards
            5: 'string', 6: 'string',   # bitboards
            7: 'string', 8: 'string',   # bitboards
            9: 'string', 10: 'string',  # bitboards
            11: 'string', 12: 'string', # bitboards
            13: 'uint8',                # side to move: 0 = white, 1 = black
            14: 'uint8',                # enpassant square: 0-63, 65 is none 
            15: 'uint8',                # castling status: integer [0000 b4 b3 b2 b1] b1 = K, b2 = Q, b3 = k, b4 = q
            16: 'int32'                 # score: integer value (mate = 2^15 - 1)
            }
    df = pd.read_csv(file, header=None, dtype=dtypes)

    fens = df.iloc[:, 0].values
    bitboards = df.iloc[:, 1:13].values
    aux = df.iloc[:, 13:-1].values
    score = df.iloc[:, -1:].values

    x = [(np.array([[string_to_matrix(b) for b in bs]])) for bs in bitboards]
    aux = [(np.array([v])) for v in aux]
    score = [(np.array([v / 100.0])) for v in score]

    return fens, x, aux, score

def rescale_bitboards(bs):
    """ Scale each bitboard by [0.1, 0.2, ..., 0.6] relatively for both colors """
    return np.array([bs[i] * (i%6 + 1) / 10 for i in np.arange(12)])

def rotate_board(bitboards, aux, score):
    """ Rotate the board 180 degrees, switching the colors of pieces, side to move, castling status and en-passant
    square """
    # Bitboards are shaped 12x8x8
    wpieces = bitboards[:6]
    bpieces = bitboards[6:]

    # Side to move is 0 if it's white's turn, 1 otherwise
    stm = (int(aux[0]) + 1) % 2                         # switch side to move

    # 0-63 for valid squares, 65 for invalid ones (64 is unused)
    ep0 = int(aux[1])
    ep = 63 - ep0 if 0 <= ep0 < 64 else ep0    # rotate en passant square

    if ep0 < 0:
        print(bitboards)
        print(aux)
        print(score)
        raise Exception('Invalid enpassant square')

    # Castling is encoded in the 4 least significant bits of a byte:
    # b4 b3 b2 b1 -> We swap b4 with b2 and b3 with b1 (0xC = 1100, 0x3 = 0011).
    castle = (int(aux[2]) & 0xC) >> 2 | (int(aux[2]) & 0x3) << 2 # swap castling white <-> castling black

    return torch.cat([bpieces, wpieces], dim=0).flip(-2, -1), torch.tensor([stm, ep, castle]).float(), -score
