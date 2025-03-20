import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Dict, List, Optional, Tuple, Union, Any
import logging
import os
import time

class QuantizationConfig:
    """Configuration for quantization of SSM models."""
    
    def __init__(
        self,
        bits: int = 8,
        group_size: int = 128,
        sym: bool = True,
        per_token: bool = False,
        per_channel: bool = True,
        static_groups: bool = False,
        use_cuda_kernels: bool = True,
    ):
        """
        Initialize quantization configuration.
        
        Args:
            bits: Bit width for quantization (4, 8, or 16)
            group_size: Size of groups for group-wise quantization 
            sym: Whether to use symmetric quantization
            per_token: Whether to quantize per token
            per_channel: Whether to quantize per channel
            static_groups: Whether to use static groups
            use_cuda_kernels: Whether to use custom CUDA kernels for quantized operations
        """
        self.bits = bits
        self.group_size = group_size
        self.sym = sym
        self.per_token = per_token
        self.per_channel = per_channel
        self.static_groups = static_groups
        self.use_cuda_kernels = use_cuda_kernels and torch.cuda.is_available()
        
        # Validate configuration
        if bits not in [4, 8, 16]:
            raise ValueError(f"Unsupported bit width: {bits}. Supported values are 4, 8, and 16.")
        
        if bits == 4 and group_size <= 0:
            raise ValueError("Group size must be positive for 4-bit quantization.")
        
        self.dtype = {
            4: torch.uint8,
            8: torch.uint8,
            16: torch.float16
        }[bits]
        
        logging.info(f"Initializing quantization with config: {self.__dict__}")

def quantize_tensor(tensor: torch.Tensor, config: QuantizationConfig) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """
    Quantize a tensor according to the given configuration.
    
    Args:
        tensor: Tensor to quantize
        config: Quantization configuration
        
    Returns:
        Tuple of (quantized_tensor, scales, zero_points)
    """
    org_device = tensor.device
    if config.use_cuda_kernels and org_device.type == "cuda":
        try:
            # Try to use custom CUDA kernels if available
            from edge.kernels.quantization import cuda_quantize
            return cuda_quantize(tensor, config.bits, config.group_size, config.sym)
        except (ImportError, RuntimeError) as e:
            logging.warning(f"Failed to use CUDA kernels for quantization: {e}. Falling back to PyTorch implementation.")
    
    # PyTorch implementation
    org_shape = tensor.shape
    tensor = tensor.to(torch.float32).detach()
    
    if config.bits == 16:
        # For 16-bit, we can simply use float16
        return tensor.to(torch.float16), None, None
    
    # Reshape for group quantization if needed
    if config.group_size > 0 and tensor.numel() > config.group_size:
        if config.per_channel:
            # Group per channel (typically for weights)
            group_shape = (-1, config.group_size)
            tensor = tensor.reshape(group_shape)
        elif config.per_token:
            # Group per token (typically for activations)
            if len(org_shape) == 3:  # (batch, seq_len, dim)
                tensor = tensor.reshape(org_shape[0], org_shape[1], -1, config.group_size)
            else:
                tensor = tensor.reshape(-1, config.group_size)
    
    # Calculate scale and zero point
    if config.sym:
        # Symmetric quantization
        max_abs = torch.max(torch.abs(tensor), dim=-1, keepdim=True)[0]
        scale = max_abs / (2**(config.bits-1) - 1)
        zero_point = torch.zeros_like(scale, dtype=torch.int)
    else:
        # Asymmetric quantization
        min_val = torch.min(tensor, dim=-1, keepdim=True)[0]
        max_val = torch.max(tensor, dim=-1, keepdim=True)[0]
        scale = (max_val - min_val) / (2**config.bits - 1)
        zero_point = torch.round(-min_val / scale).clamp(0, 2**config.bits - 1).to(torch.int)
    
    # Prevent division by zero
    scale = torch.max(scale, torch.full_like(scale, 1e-10))
    
    # Quantize
    qvalue = torch.round(tensor / scale + zero_point if not config.sym else tensor / scale)
    
    # Clamp to ensure values are within range
    if config.sym:
        qvalue = qvalue.clamp(-(2**(config.bits-1)), 2**(config.bits-1) - 1)
    else:
        qvalue = qvalue.clamp(0, 2**config.bits - 1)
    
    # Pack values if bits < 8
    if config.bits == 4:
        # Pack two 4-bit values into one 8-bit value
        qvalue = qvalue.reshape(*qvalue.shape[:-1], -1, 2)
        qvalue_packed = qvalue[..., 0] | (qvalue[..., 1] << 4)
        qvalue = qvalue_packed
    
    # Return quantized tensor and quantization parameters
    return qvalue.to(config.dtype).to(org_device), scale.to(org_device), zero_point.to(org_device)

def dequantize_tensor(qvalue: torch.Tensor, scale: torch.Tensor, zero_point: torch.Tensor, 
                      config: QuantizationConfig, original_shape: Optional[tuple] = None) -> torch.Tensor:
    """
    Dequantize a tensor according to the given configuration.
    
    Args:
        qvalue: Quantized tensor
        scale: Scale factor
        zero_point: Zero point
        config: Quantization configuration
        original_shape: Original shape of the tensor before quantization
        
    Returns:
        Dequantized tensor
    """
    if config.bits == 16:
        return qvalue
    
    org_device = qvalue.device
    if config.use_cuda_kernels and org_device.type == "cuda":
        try:
            from edge.kernels.quantization import cuda_dequantize
            return cuda_dequantize(qvalue, scale, zero_point, config.bits)
        except (ImportError, RuntimeError) as e:
            logging.warning(f"Failed to use CUDA kernels for dequantization: {e}. Falling back to PyTorch implementation.")
    
    # Unpack values if needed
    if config.bits == 4:
        unpacked = torch.zeros((*qvalue.shape, 2), dtype=torch.int8, device=qvalue.device)
        unpacked[..., 0] = qvalue & 0xF
        unpacked[..., 1] = (qvalue >> 4) & 0xF
        # Sign extend if symmetric
        if config.sym:
            mask = 0x8
            unpacked[..., 0] = unpacked[..., 0] - ((unpacked[..., 0] & mask) * 2)
            unpacked[..., 1] = unpacked[..., 1] - ((unpacked[..., 1] & mask) * 2)
        qvalue = unpacked.reshape(*qvalue.shape[:-1], -1)
    
    # Convert to float and dequantize
    qvalue = qvalue.to(torch.float32)
    dequantized = qvalue * scale if config.sym else (qvalue - zero_point) * scale
    
    # Reshape to original shape if provided
    if original_shape is not None:
        dequantized = dequantized.reshape(original_shape)
    
    return dequantized

class QuantizedLinear(nn.Module):
    """Quantized linear layer for SSM models."""
    
    def __init__(
        self,
        linear: nn.Linear,
        config: QuantizationConfig,
    ):
        """
        Initialize a quantized linear layer.
        
        Args:
            linear: Original linear layer
            config: Quantization configuration
        """
        super().__init__()
        self.config = config
        self.in_features = linear.in_features
        self.out_features = linear.out_features
        
        # Quantize weights
        q_weight, self.weight_scale, self.weight_zero_point = quantize_tensor(
            linear.weight.data, config
        )
        self.register_buffer('q_weight', q_weight)
        self.register_buffer('weight_scale', self.weight_scale)
        if not config.sym:
            self.register_buffer('weight_zero_point', self.weight_zero_point)
        
        # Handle bias
        if linear.bias is not None:
            self.bias = nn.Parameter(linear.bias.data)
        else:
            self.register_parameter('bias', None)
        
        # For activation quantization (used in forward pass)
        self.act_scale = None
        self.act_zero_point = None
        
        # Original shape for unpacking
        self.orig_weight_shape = linear.weight.shape
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass with quantized weights."""
        # Quantize input if needed for full int8 computation
        if self.training or self.act_scale is None:
            # During training or first inference, compute activation stats
            act_config = QuantizationConfig(
                bits=8, 
                sym=self.config.sym,
                per_token=True,
                per_channel=False,
                group_size=self.config.group_size,
                use_cuda_kernels=self.config.use_cuda_kernels
            )
            q_input, act_scale, act_zero_point = quantize_tensor(x, act_config)
            if not self.training:
                # Cache for future inference
                self.act_scale = act_scale
                if not act_config.sym:
                    self.act_zero_point = act_zero_point
        else:
            # Use cached activation stats for faster inference
            act_config = QuantizationConfig(
                bits=8, 
                sym=self.config.sym,
                per_token=True,
                per_channel=False,
                group_size=self.config.group_size,
                use_cuda_kernels=self.config.use_cuda_kernels
            )
            q_input, _, _ = quantize_tensor(x, act_config)
        
        # Dequantize weights for computation
        weight = dequantize_tensor(
            self.q_weight, self.weight_scale, 
            getattr(self, 'weight_zero_point', None), 
            self.config, self.orig_weight_shape
        )
        
        # Perform the linear operation
        output = F.linear(x, weight, self.bias)
        
        return output

class QuantizedSSMLayer(nn.Module):
    """Wrapper for quantizing SSM layers."""
    
    def __init__(self, module: nn.Module, config: QuantizationConfig):
        """
        Initialize a quantized SSM layer.
        
        Args:
            module: Original SSM layer
            config: Quantization configuration
        """
        super().__init__()
        self.module = module
        self.config = config
        
        # Replace linear layers with quantized versions
        self._replace_linears(self.module, config)
    
    def _replace_linears(self, module: nn.Module, config: QuantizationConfig, prefix: str = ''):
        """
        Recursively replace linear layers with quantized versions.
        
        Args:
            module: Module to process
            config: Quantization configuration
            prefix: Prefix for logging
        """
        for name, child in list(module.named_children()):
            full_name = f"{prefix}.{name}" if prefix else name
            
            if isinstance(child, nn.Linear):
                setattr(module, name, QuantizedLinear(child, config))
                logging.info(f"Quantized linear layer: {full_name}")
            else:
                self._replace_linears(child, config, full_name)
    
    def forward(self, *args, **kwargs):
        """Forward pass using the quantized module."""
        return self.module(*args, **kwargs)

class ModelQuantizer:
    """Utility class for quantizing SSM models."""
    
    @staticmethod
    def quantize_model(model: nn.Module, config: QuantizationConfig) -> nn.Module:
        """
        Quantize an entire model.
        
        Args:
            model: Model to quantize
            config: Quantization configuration
            
        Returns:
            Quantized model
        """
        # Make a deepcopy to avoid modifying the original model
        model = model.eval()  # Set to eval mode for stable quantization
        
        # Start by identifying and replacing SSM layers
        for name, module in list(model.named_children()):
            # Check if this is an SSM layer - look for specific attributes that indicate SSM
            # This will need to be adapted based on the specific SSM implementation
            is_ssm = hasattr(module, 'A') or hasattr(module, 'B') or hasattr(module, 'C') or \
                    hasattr(module, 'dt') or 'SSM' in module.__class__.__name__ or \
                    'Mamba' in module.__class__.__name__ or 'S6' in module.__class__.__name__
            
            if is_ssm:
                # This is an SSM layer, wrap it with our quantized version
                setattr(model, name, QuantizedSSMLayer(module, config))
            elif len(list(module.children())) > 0:
                # This is a container module, recurse into it
                setattr(model, name, ModelQuantizer.quantize_model(module, config))
        
        return model
    
    @staticmethod
    def calibrate_model(model: nn.Module, calib_data_loader, 
                        num_batches: int = 10) -> nn.Module:
        """
        Calibrate the quantized model using representative data.
        
        Args:
            model: Quantized model
            calib_data_loader: DataLoader with calibration data
            num_batches: Number of batches to use for calibration
            
        Returns:
            Calibrated model
        """
        model.eval()
        with torch.no_grad():
            for i, batch in enumerate(calib_data_loader):
                if i >= num_batches:
                    break
                
                # Process input data according to your model's expected format
                if isinstance(batch, (list, tuple)):
                    inputs = batch[0].to(next(model.parameters()).device)
                else:
                    inputs = batch.to(next(model.parameters()).device)
                
                # Just run a forward pass to populate activation stats
                _ = model(inputs)
                
                logging.info(f"Calibration: batch {i+1}/{num_batches} processed")
        
        return model
    
    @staticmethod
    def export_model(model: nn.Module, output_path: str, 
                     metadata: Optional[Dict[str, Any]] = None) -> None:
        """
        Export the quantized model.
        
        Args:
            model: Quantized model
            output_path: Path to save the model
            metadata: Optional metadata to save with the model
        """
        os.makedirs(os.path.dirname(output_path), exist_ok=True)
        
        export_dict = {
            'model_state_dict': model.state_dict(),
            'quantization_info': {
                'version': '1.0',
                'timestamp': time.strftime('%Y-%m-%d %H:%M:%S'),
            }
        }
        
        if metadata:
            export_dict['metadata'] = metadata
        
        torch.save(export_dict, output_path)
        logging.info(f"Quantized model exported to {output_path}")

def benchmark_model(model: nn.Module, input_size: tuple, 
                    device: torch.device, num_runs: int = 100):
    """
    Benchmark a model's inference speed and memory usage.
    
    Args:
        model: Model to benchmark
        input_size: Size of input tensor
        device: Device to run on
        num_runs: Number of runs for averaging
        
    Returns:
        Dict with benchmark results
    """
    model = model.to(device).eval()
    x = torch.randn(input_size).to(device)
    
    # Warmup
    with torch.no_grad():
        for _ in range(10):
            _ = model(x)
    
    # Time measurement
    torch.cuda.synchronize() if device.type == 'cuda' else None
    start_time = time.time()
    
    with torch.no_grad():
        for _ in range(num_runs):
            _ = model(x)
    
    torch.cuda.synchronize() if device.type == 'cuda' else None
    end_time = time.time()
    
    # Memory usage (for CUDA only)
    if device.type == 'cuda':
        torch.cuda.reset_peak_memory_stats()
        with torch.no_grad():
            _ = model(x)
        memory_allocated = torch.cuda.max_memory_allocated() / (1024 ** 2)  # MB
    else:
        memory_allocated = 0
    
    avg_time = (end_time - start_time) / num_runs
    return {
        'latency_ms': avg_time * 1000,
        'throughput_samples_per_sec': 1 / avg_time * input_size[0],
        'memory_usage_mb': memory_allocated,
        'device': str(device),
        'precision': model.__class__.__name__
    }

def quantize_model_from_name(model_name: str, config: QuantizationConfig) -> nn.Module:
    """
    Load and quantize a model by name.
    
    Args:
        model_name: Name of the model to load and quantize
        config: Quantization configuration
        
    Returns:
        Quantized model
    """
    from cartesia_pytorch.Llamba.llamba import LlambaLMHeadModel
    from cartesia_pytorch.Rene.rene import ReneLMHeadModel
    
    # Load model based on name
    if model_name == "Llamba-1B":
        model = LlambaLMHeadModel.from_pretrained("cartesia-ai/Llamba-1B")
    elif model_name == "Llamba-3B":
        model = LlambaLMHeadModel.from_pretrained("cartesia-ai/Llamba-3B")
    elif model_name == "Llamba-8B":
        model = LlambaLMHeadModel.from_pretrained("cartesia-ai/Llamba-8B")
    elif model_name == "Rene":
        model = ReneLMHeadModel.from_pretrained("cartesia-ai/Rene-v0.1-1.3b-pytorch")
    else:
        raise ValueError(f"Unknown model name: {model_name}")
    
    # Quantize the model
    return ModelQuantizer.quantize_model(model, config)

def main():
    """Example usage of quantization functions."""
    import argparse
    
    parser = argparse.ArgumentParser(description='SSM Model Quantization')
    parser.add_argument('--model_path', type=str, required=True, help='Path to the pre-trained model')
    parser.add_argument('--output_path', type=str, required=True, help='Path to save the quantized model')
    parser.add_argument('--bits', type=int, default=8, choices=[4, 8, 16], help='Quantization bit width')
    parser.add_argument('--group_size', type=int, default=128, help='Group size for quantization')
    parser.add_argument('--symmetric', action='store_true', help='Use symmetric quantization')
    parser.add_argument('--benchmark', action='store_true', help='Run benchmark after quantization')
    parser.add_argument('--device', type=str, default='cuda' if torch.cuda.is_available() else 'cpu',
                       help='Device to run on')
    
    args = parser.parse_args()
    
    # Set up logging
    logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
    
    # Load model
    logging.info(f"Loading model from {args.model_path}")
    model = torch.load(args.model_path)
    
    # Set up quantization configuration
    config = QuantizationConfig(
        bits=args.bits,
        group_size=args.group_size,
        sym=args.symmetric,
        per_channel=True,
        use_cuda_kernels=args.device == 'cuda'
    )
    
    # Quantize model
    logging.info(f"Quantizing model to {args.bits} bits")
    quantized_model = ModelQuantizer.quantize_model(model, config)
    
    # Export model
    logging.info(f"Exporting quantized model to {args.output_path}")
    ModelQuantizer.export_model(quantized_model, args.output_path, {
        'original_model': args.model_path,
        'quantization_config': vars(config)
    })
    
    # Benchmark if requested
    if args.benchmark:
        logging.info("Running benchmark")
        device = torch.device(args.device)
        
        # Assuming a standard input size, adjust as needed
        input_size = (1, 512, model.config.hidden_size if hasattr(model, 'config') else 768)
        
        # Benchmark original model
        orig_results = benchmark_model(model, input_size, device)
        logging.info(f"Original model performance: {orig_results}")
        
        # Benchmark quantized model
        quant_results = benchmark_model(quantized_model, input_size, device)
        logging.info(f"Quantized model performance: {quant_results}")
        
        # Print comparison
        speedup = orig_results['latency_ms'] / quant_results['latency_ms']
        memory_reduction = orig_results['memory_usage_mb'] / (quant_results['memory_usage_mb'] + 1e-8)
        
        logging.info(f"Speedup: {speedup:.2f}x")
        logging.info(f"Memory reduction: {memory_reduction:.2f}x")

if __name__ == '__main__':
    main()