import numpy

from vm_compiler.addressing import build_ring_address_keys, build_ring_address_record
from vm_compiler.fp4 import decode_e2m1
from vm_compiler.hullkv import build_hull_2d, support_topk_2d


def test_every_ring_key_is_an_exact_stable_2d_attention_address():
    keys = build_ring_address_keys(128)
    hull = build_hull_2d(keys)
    winners = support_topk_2d(keys, keys, hull, k=1)

    assert hull.tolist() == list(range(128))
    assert winners[:, 0].tolist() == list(range(128))


def test_per_element_scaled_fp4_preserves_ring_keys_exactly():
    keys = build_ring_address_keys(128)
    record = build_ring_address_record("vocabulary_addresses", 128)
    lattice = decode_e2m1(numpy.frombuffer(record.packed, dtype=numpy.uint8), keys.size)
    decoded = (lattice * numpy.asarray(record.scales, dtype=numpy.float32)).reshape(128, 2)

    assert record.shape == (128, 2)
    assert record.block_size == 1
    assert numpy.array_equal(decoded, keys)
