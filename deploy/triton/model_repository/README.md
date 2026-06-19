# Triton model repository

Drop exported engines here in Triton's expected layout, one dir per model:

```
model_repository/
  person_detect/
    config.pbtxt
    1/model.plan          # TensorRT engine (or model.onnx)
  pose/
    config.pbtxt
    1/model.plan
  parsing/
    config.pbtxt
    1/model.onnx
  garment/
    config.pbtxt
    1/model.plan
```

Example `config.pbtxt` (person detector, dynamic batching):

```
name: "person_detect"
platform: "tensorrt_plan"
max_batch_size: 16
input  [ { name: "images", data_type: TYPE_FP16, dims: [3, 640, 640] } ]
output [ { name: "output0", data_type: TYPE_FP16, dims: [-1, -1] } ]
dynamic_batching { preferred_batch_size: [4, 8, 16] max_queue_delay_microseconds: 2000 }
instance_group [ { count: 1, kind: KIND_GPU } ]
```

Serve:
```bash
tritonserver --model-repository=deploy/triton/model_repository
```
The Clothic AI perception backends call Triton over gRPC; internal services use
gRPC, external clients use the REST/WebSocket API in `clothic.api`.
```
