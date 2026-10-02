Release the initial ONNX runtime before constructing the fixed-batch compatibility runtime to avoid overlapping model allocations when finalizing neural drives.
