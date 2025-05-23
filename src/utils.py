import wandb
import torch
import os
import torch.onnx

def download_wandb_checkpoint(run_path, filename, device='cuda'):
    api = wandb.Api()

    run = api.run(run_path)
    run.file(filename).download(replace=True)
    checkpoint = torch.load(filename, map_location=torch.device(device))
    return checkpoint

def save_wandb_file(path):
    wandb.save(path, base_path=os.path.dirname(path))

def export_onnx(model):
    # --- Configuration ---
    MODEL_NAME = "resnet18" # Or any other model like "mobilenet_v2", etc.
    OUTPUT_ONNX_FILE = "model.onnx"
    # Define input shape (Batch Size, Channels, Height, Width)
    # Use a fixed batch size (e.g., 1) for the dummy input if your C++ code assumes it
    BATCH_SIZE = 1
    INPUT_HEIGHT = 8
    INPUT_WIDTH = 8
    INPUT_CHANNELS = 12
    # Choose an ONNX opset version. 14-17 are good modern choices.
    # Ensure your ONNX Runtime version supports this opset.
    OPSET_VERSION = 17
    # Define names for input/output nodes - MUST match C++ expectations (if not read dynamically)
    # --- End Configuration ---

    # Set the model to evaluation mode (important for layers like BatchNorm, Dropout)
    model.eval()

    # Create a dummy input tensor with the correct shape and type
    dummy_input = torch.randn(BATCH_SIZE, INPUT_CHANNELS, INPUT_HEIGHT, INPUT_WIDTH, requires_grad=False)
    dummy_input = (dummy_input, torch.randn(BATCH_SIZE, 3)) # Add auxiliary inputs (side, ep, castling)

    print(f"Exporting model to ONNX file: {OUTPUT_ONNX_FILE}")
    print(f"  Opset Version: {OPSET_VERSION}")

    try:
        # Export the model
        torch.onnx.export(
            model,                     # Model being run
            dummy_input,               # Model input (or tuple for multiple inputs)
            OUTPUT_ONNX_FILE,          # Where to save the model
            export_params=True,        # Store the trained parameter weights inside the model file
            opset_version=OPSET_VERSION, # ONNX version to export the model to
            do_constant_folding=True,  # Execute constant folding for optimization
            input_names=["bitboards", "aux"],  # Model's input names
            output_names=["score"], # Model's output names
            # dynamic_axes={'input' : {0 : 'batch_size'},    # Variable length axes
            #               'output' : {0 : 'batch_size'}}  # Uncomment if batch size needs to be dynamic
        )
        print("-" * 30)
        print(f"Model successfully exported to {os.path.abspath(OUTPUT_ONNX_FILE)}")
        print("-" * 30)
        print("You can now use this `.onnx` file with the C++ ONNX Runtime example.")

    except Exception as e:
        print(f"\nError during ONNX export: {e}")
        # Add more specific error handling if needed
