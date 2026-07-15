# Model

The installed model `nanodet-plus-m_320.onnx` is the official
NanoDet-Plus-m 320 x 320 COCO model from the RangiLyu/nanodet
v1.0.0-alpha-1 GitHub release. NanoDet is licensed under Apache-2.0.

Source URL:

`https://github.com/RangiLyu/nanodet/releases/download/v1.0.0-alpha-1/nanodet-plus-m_320.onnx`

The bundled graph has legacy initializers removed from its input list with
ONNX Runtime's official `remove_initializer_from_input` utility. This does not
change its inference output and allows current runtimes to optimize the graph.

Cleaned model SHA-256:

`95690e7d59ecb20c57d614f33fd54ab2f84fcceb9002e707c5c85f298006b480`
