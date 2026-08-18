#!/usr/bin/env python3
import unittest

import torch

from glm_dsa_trim_dump import build_semantic_artifact


class SemanticArtifactTest(unittest.TestCase):
    def test_normalizes_prefill_boundaries_for_sglang_contract(self):
        dump = {
            "embed": torch.ones(1, 3, 4),
            "layer0_in": torch.full((1, 3, 4), 2.0),
            "layer0_attn": torch.full((1, 3, 4), 3.0),
            "layer0_mlp_in": torch.full((1, 3, 4), 4.0),
            "layer0_out": torch.full((1, 3, 4), 5.0),
            "layer0_dsa": {"indices": torch.arange(6).reshape(1, 3, 2)},
            "norm": torch.full((1, 3, 4), 6.0),
            "logits": torch.full((1, 1, 7), 7.0),
        }

        artifact = build_semantic_artifact({"step0": dump}, num_layers=1)

        self.assertEqual(artifact["format"], "glm52_transformers_semantic_v1")
        self.assertEqual(artifact["producer"], "transformers")
        boundaries = artifact["steps"]["prefill"]
        self.assertEqual(
            set(boundaries),
            {
                "embedding",
                "layer_input",
                "post_attention_residual",
                "mlp_input",
                "layer_output",
                "final_norm",
                "logits",
                "dsa_topk",
            },
        )
        self.assertTrue(torch.equal(boundaries["layer_output"][0], dump["layer0_out"]))
        self.assertTrue(torch.equal(boundaries["dsa_topk"][0], dump["layer0_dsa"]["indices"]))


if __name__ == "__main__":
    unittest.main()
