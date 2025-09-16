# cluelyirl
to run it in terminal, make sure to enable microphone in privacy & security

# setup
- clone stt model https://huggingface.co/csukuangfj/sherpa-onnx-nemo-parakeet-tdt-0.6b-v2


```
uv run parakeet_stt.py --energy-threshold 1e-6 --silence-ms 600 --debug
```
