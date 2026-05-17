from sageattn4 import last_fwd_used_specialized, sageattn4_blackwell
from unittest import TestCase
import unittest
import torch.nn.functional as F
import torch

class AttentionUnitTest(TestCase):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    def test_basic(self):
        B,H,T,D = 4, 8, 1024, 128

        token_tile = 112
        T = T // token_tile * token_tile

        print(f"T: {T}")

        dtype = torch.bfloat16

        embed = torch.load("fixtures/llama3p1_8b_instruct_l16_h30_s256_qkv.pt")

        print(embed.keys())

        q = embed['q'].cuda()
        k = embed['k'].cuda()
        v = embed['v'].cuda()

        print("Q=", q.shape, "K=", k.shape, "V=", v.shape)

        out = sageattn4_blackwell(q, k, v, is_causal=False, per_block_mean=True)
        torch.cuda.synchronize()
        self.assertTrue(last_fwd_used_specialized())
        ref = F.scaled_dot_product_attention(q, k, v, is_causal=False)

        err = 1-F.cosine_similarity(out, ref, dim=-1).mean()
        err_l1 = F.l1_loss(out, ref).mean()
        thresh = 0.01
        thresh_l1 = 0.01
        if err > thresh or err_l1 > thresh:
            print("==== REF ====")
            print(ref[0, 0, :8, :1])
            print("==== OUT ====")
            print(out[0, 0, :8, :1])
        else:
            print("Err=", err, "err l1=", err_l1)
        self.assertLess(err, thresh)
        self.assertLess(err_l1, thresh_l1)

if __name__ == '__main__':
    unittest.main()
