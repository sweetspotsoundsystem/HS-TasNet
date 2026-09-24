# Current model ONNX lowering

The public API is `stemgenrt.export.export_model`. It accepts the maintained
`StemgenRT58`, with its fixed 1024-sample analysis window, 256-sample
synthesis frame, 128-sample hop and the saved eight or nine explicit states. Both export variants
preserve the source model, use self-contained weights and publish only after
independent state-trajectory verification.

- `fp32.py` lowers the native model, including real FFTs and one-frame GRUs.
- `integer.py` quantizes ten encoder/fusion/mask products and retains an
  independently reconstructed NumPy/PyTorch integer reference.
- `integer_precision.py` retains FP64 quantizer ancestors, FP32 integer
  dequantization/bias blocks, FP32 decoding and public states.
- `integer_branch_gru.py` adds four branch GRU products; their biases remain
  outside the integer blocks in FP64.
- `integer_branch_output.py` adds the two bias-free branch output products.
- `integer_qkv_fusion.py` and `integer_qkv.py` pack query/key/value columns into
  the seventeenth integer product.
- `integer_qkv_reference.py` reconstructs the full integer model independently,
  verifying every quantized weight, scale and zero point against native FP32
  tensors before comparing the graph's recurrent outputs.
- `deployment.py` composes these stages for `variant="integer"`.

The lower-level rewrite functions still default to the historical authenticated
parent graph digests. The maintained pipeline explicitly supplies the digest of
each preceding in-memory stage, preserving the topology, shape and matrix
identity checks. Source-path metadata changes on extraction, so a newly built
artifact is identified by its own digest; it is not asserted to be byte-identical
to a historical release.

The sixteen-/seventeen-product rewrites support the current lowering's specific
node inventory. They fail if that inventory changes rather than silently
quantizing a different network. FP32 and integer candidates need their own
quality evaluation and host-timing measurements. Numerical verification does
not supply either measurement.
