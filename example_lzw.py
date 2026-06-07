import numpy as np

def lzw_encode(data: np.ndarray) -> list[int]:
    # Initialize dictionary with all single values (0-255)
    dictionary = {(i,): i for i in range(256)}
    next_code = 256

    w = ()
    output = []

    for val in data:
        wc = w + (int(val),)
        if wc in dictionary:
            w = wc
        else:
            output.append(dictionary[w])
            dictionary[wc] = next_code
            next_code += 1
            w = (int(val),)

    if w:
        output.append(dictionary[w])

    return output


def lzw_decode(codes: list[int], dtype=np.uint8) -> np.ndarray:
    # Initialize dictionary with single values
    dictionary = {i: (i,) for i in range(256)}
    next_code = 256

    result = []
    w = dictionary[codes[0]]
    result.extend(w)

    for code in codes[1:]:
        if code in dictionary:
            entry = dictionary[code]
        elif code == next_code:
            entry = w + (w[0],)  # Special case
        else:
            raise ValueError(f"Bad code: {code}")

        result.extend(entry)
        dictionary[next_code] = w + (entry[0],)
        next_code += 1
        w = entry

    return np.array(result, dtype=dtype)


# --- Test ---
original = np.array([10, 20, 10, 20, 10, 20, 10, 20, 30, 30, 30, 30, 10, 20, 10, 20, 10, 20, 10, 20, 30, 30, 30, 30, 4, 10, 20, 10, 20, 10, 20, 10, 20, 30, 30, 30, 30, 10, 20, 10, 20, 10, 20, 10, 20, 30, 30, 30, 30], dtype=np.uint8)
encoded = lzw_encode(original)
decoded = lzw_decode(encoded)

print(f"Original:  {original} Length: {len(original)}")
print(f"Encoded:   {encoded} Length: {len(encoded)}")
print(f"Decoded:   {decoded} Length: {len(decoded)}")
print(f"Match:     {np.array_equal(original, decoded)}")