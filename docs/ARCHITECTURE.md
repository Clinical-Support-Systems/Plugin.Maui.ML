# Plugin.Maui.ML - Architecture Overview

## Visual Architecture

```mermaid
graph TB
    subgraph "Application Layer"
        APP["Your .NET MAUI App"]
    end
    
    subgraph "Plugin Interface"
        IMLIF["IMLInfer Interface"]
        PLUGIN["MLPlugin.Default"]
    end
    
    subgraph "Backend Implementations"
        ONNX["OnnxRuntimeInfer<br/>(Cross-Platform)"]
        COREML["CoreMLInfer<br/>(iOS/macOS Native)"]
        MLKIT["MLKitInfer<br/>(Android - Future)"]
        WINML["WindowsMLInfer<br/>(Windows - Future)"]
    end
    
    subgraph "Platform Wrappers"
        PIOS["PlatformMLInfer<br/>(iOS)"]
        PANDROID["PlatformMLInfer<br/>(Android)"]
        PWIN["PlatformMLInfer<br/>(Windows)"]
        PMAC["PlatformMLInfer<br/>(macOS)"]
    end
    
    subgraph "Hardware Acceleration"
        NE["Apple Neural Engine"]
        NNAPI["Android NNAPI"]
        DML["Windows DirectML"]
        CPU["CPU Fallback"]
    end
    
    APP --> IMLIF
    APP --> PLUGIN
    
    PLUGIN -.->|"Auto-select"| PIOS
    PLUGIN -.->|"Auto-select"| PANDROID
    PLUGIN -.->|"Auto-select"| PWIN
    PLUGIN -.->|"Auto-select"| PMAC
    
    IMLIF --> ONNX
    IMLIF --> COREML
    IMLIF --> MLKIT
    IMLIF --> WINML
    
    PIOS --> ONNX
    PANDROID --> ONNX
    PWIN --> ONNX
    PMAC --> ONNX
    
    ONNX -.->|"CoreML EP"| NE
    ONNX -.->|"NNAPI EP"| NNAPI
    ONNX -.->|"DirectML EP"| DML
    ONNX --> CPU
    
    COREML --> NE
    COREML --> CPU
    
    MLKIT -.-> NNAPI
    MLKIT -.-> CPU
    
    WINML -.-> DML
    WINML -.-> CPU
    
    style APP fill:#E3F2FD,stroke:#1976D2
    style IMLIF fill:#4CAF50,stroke:#2E7D32,color:#fff
    style PLUGIN fill:#4CAF50,stroke:#2E7D32,color:#fff
    style ONNX fill:#2196F3,stroke:#1565C0,color:#fff
    style COREML fill:#FF9800,stroke:#E65100,color:#fff
    style MLKIT fill:#9C27B0,stroke:#4A148C,color:#fff
    style WINML fill:#00BCD4,stroke:#006064,color:#fff
    style NE fill:#FFE0B2,stroke:#E65100
    style NNAPI fill:#E1BEE7,stroke:#4A148C
    style DML fill:#B2EBF2,stroke:#006064
    style CPU fill:#CFD8DC,stroke:#455A64
```

## Component Breakdown

### 1. Application Layer
Your .NET MAUI application that consumes the ML inference services.

```csharp
public class MyApp
{
    private readonly IMLInfer _mlService;
    
    public MyApp(IMLInfer mlService)
    {
        _mlService = mlService; // Injected by DI
    }
}
```

### 2. Plugin Interface Layer
Provides unified interface and automatic backend selection.

```csharp
// Interface
public interface IMLInfer
{
    MLBackend Backend { get; }
    Task<Dictionary<string, Tensor<float>>> RunInferenceAsync(...);
    // ...
}

// Auto-selection
var mlService = MLPlugin.Default; // Automatically picks best backend
```

### 3. Backend Implementations

#### ONNX Runtime (Default)
```csharp
public class OnnxRuntimeInfer : IMLInfer
{
    public MLBackend Backend => MLBackend.OnnxRuntime;
    // Uses ONNX format (.onnx files)
    // Works on all platforms
}
```

#### CoreML (iOS/macOS)
```csharp
public class CoreMLInfer : IMLInfer
{
    public MLBackend Backend => MLBackend.CoreML;
    // Uses CoreML format (.mlmodel, .mlmodelc)
    // iOS/macOS only
    // Direct Neural Engine access
}
```

### 4. Platform Wrappers
Provide platform-specific optimizations while using base implementations.

```csharp
// iOS Platform Wrapper
public class PlatformMLInfer : OnnxRuntimeInfer
{
    // Inherits ONNX Runtime
    // Adds CoreML execution provider
    // Provides iOS-specific helpers
    
    public static IMLInfer CreateCoreMLInfer()
    {
        return new CoreMLInfer(); // Factory for pure CoreML
    }
}
```

### 5. Hardware Acceleration

| Platform | Primary | Secondary | Fallback |
|----------|---------|-----------|----------|
| **iOS/macOS** | Neural Engine | CoreML | CPU |
| **Android** | NPU via NNAPI | GPU | CPU |
| **Windows** | GPU via DirectML | - | CPU |

## Data Flow Examples

### Example 1: Automatic Backend Selection
```mermaid
sequenceDiagram
    participant App
    participant MLPlugin
    participant PlatformInfer
    participant Hardware
    
    App->>MLPlugin: Request MLPlugin.Default
    MLPlugin->>MLPlugin: Detect platform
    
    alt iOS/macOS
        MLPlugin->>PlatformInfer: Create iOS PlatformMLInfer
        PlatformInfer->>Hardware: Configure CoreML EP
    else Android
        MLPlugin->>PlatformInfer: Create Android PlatformMLInfer
        PlatformInfer->>Hardware: Configure NNAPI EP
    else Windows
        MLPlugin->>PlatformInfer: Create Windows PlatformMLInfer
        PlatformInfer->>Hardware: Configure DirectML EP
    end
    
    MLPlugin-->>App: Return configured IMLInfer
```

### Example 2: Inference Execution
```mermaid
sequenceDiagram
    participant App
    participant IMLInfer
    participant Backend
    participant HW
    
    App->>IMLInfer: LoadModelAsync("model.onnx")
    IMLInfer->>Backend: Load and compile model
    Backend->>HW: Optimize for hardware
    HW-->>Backend: Ready
    Backend-->>IMLInfer: Model loaded
    IMLInfer-->>App: Success
    
    App->>IMLInfer: RunInferenceAsync(inputs)
    IMLInfer->>Backend: Execute inference
    Backend->>HW: Run on accelerator
    HW-->>Backend: Results
    Backend-->>IMLInfer: Output tensors
    IMLInfer-->>App: Dictionary<string, Tensor<float>>
```

## Backend Comparison Matrix

| Feature | ONNX Runtime | CoreML | ML Kit | Windows ML |
|---------|--------------|--------|--------|------------|
| **Cross-Platform** | ? Yes | ? No | ? No | ? No |
| **Model Format** | .onnx | .mlmodel | .tflite | .onnx |
| **Platforms** | All | iOS/macOS | Android | Windows |
| **Status** | ? Stable | ? Stable | ?? Planned | ?? Planned |
| **Hardware Accel** | All | Neural Engine | NNAPI | DirectML |
| **Model Zoo** | Huge | Medium | Large | Medium |
| **Conversion Tools** | Many | coremltools | tf2lite | tf2onnx |
| **Memory Usage** | Medium | Low | Medium | Medium |
| **Performance** | High | Highest* | High | High |

*When using Native CoreML on Apple Silicon

## Decision Tree: Which Backend to Use?

```mermaid
graph TD
    START([Need ML Inference])
    
    START --> Q1{Single Platform?}
    
    Q1 -->|No - Multi-platform| ONNX[Use ONNX Runtime]
    Q1 -->|Yes| Q2{Which Platform?}
    
    Q2 -->|iOS/macOS| Q3{Have .mlmodel?}
    Q2 -->|Android| Q4{Have .tflite?}
    Q2 -->|Windows| Q5{Have .onnx?}
    
    Q3 -->|Yes| COREML[Use CoreMLInfer]
    Q3 -->|No| Q6{Max Performance?}
    
    Q6 -->|Yes, convert model| COREML
    Q6 -->|No, use ONNX| ONNX_IOS[Use ONNX + CoreML EP]
    
    Q4 -->|Yes| MLKIT[Use MLKitInfer<br/>Coming Soon]
    Q4 -->|No| ONNX_ANDROID[Use ONNX + NNAPI EP]
    
    Q5 -->|Yes| ONNX_WIN[Use ONNX + DirectML EP]
    Q5 -->|No, convert model| ONNX_WIN
    
    style START fill:#E3F2FD
    style ONNX fill:#2196F3,color:#fff
    style COREML fill:#FF9800,color:#fff
    style MLKIT fill:#9C27B0,color:#fff
    style ONNX_IOS fill:#64B5F6,color:#fff
    style ONNX_ANDROID fill:#64B5F6,color:#fff
    style ONNX_WIN fill:#64B5F6,color:#fff
```

## Usage Patterns

### Pattern 1: Simple (Recommended for Most Apps)
```csharp
// MauiProgram.cs
builder.Services.AddMauiML(); // Auto-optimized for platform

// Your code
public MyViewModel(IMLInfer mlService)
{
    _mlService = mlService; // Works everywhere!
}
```

### Pattern 2: Platform-Specific Optimization
```csharp
#if IOS || MACCATALYST
builder.Services.AddMauiML(MLBackend.CoreML); // Native CoreML
#elif ANDROID
builder.Services.AddMauiML(MLBackend.OnnxRuntime); // ONNX + NNAPI
#else
builder.Services.AddMauiML(); // Default
#endif
```

### Pattern 3: Runtime Backend Selection
```csharp
public class AdvancedMLService
{
    private IMLInfer _fastBackend;
    private IMLInfer _accurateBackend;
    
    public AdvancedMLService()
    {
        // Fast backend for real-time
        _fastBackend = new OnnxRuntimeInfer();
        
#if IOS || MACCATALYST
        // Accurate backend for batch processing
        _accurateBackend = new CoreMLInfer();
#else
        _accurateBackend = _fastBackend;
#endif
    }
    
    public Task<T> ProcessRealtime<T>(byte[] data)
        => RunInference(_fastBackend, data);
    
    public Task<T> ProcessBatch<T>(byte[] data)
        => RunInference(_accurateBackend, data);
}
```

## Performance Characteristics

### Inference Speed Comparison (Relative)
```
Model: ResNet50, Input: 224x224x3, Device: iPhone 14 Pro

?????????????????????????????????????????
? Backend         ? Speed    ? Memory   ?
?????????????????????????????????????????
? CoreML (Native) ? ???????? ? ??       ? Fastest, lowest memory
? ONNX + CoreML   ? ???????  ? ???      ? Very fast, good memory
? ONNX CPU        ? ????     ? ????     ? Slower, higher memory
?????????????????????????????????????????
```

### Trade-offs

**ONNX Runtime:**
- ? Works everywhere (one model, all platforms)
- ? Huge model ecosystem (PyTorch, TensorFlow, etc.)
- ? Active development
- ?? Slightly larger app size
- ?? Conversion overhead (native ? ONNX ? native)

**CoreML Native:**
- ? Best performance on Apple devices
- ? Lowest memory footprint
- ? Direct Neural Engine access
- ? iOS/macOS only
- ? Requires model conversion
- ? Smaller model ecosystem

**ML Kit (Future):**
- ? Google's high-level APIs
- ? Pre-built models for common tasks
- ? Good Android optimization
- ? Android only
- ? Less flexibility than TFLite

## Migration Path

### From ONNX-Only ? Multi-Backend

**Phase 1: No Changes Required** ?
```csharp
// Existing code continues to work
var inferService = new OnnxRuntimeInfer();
```

**Phase 2: Opt-in to Platform Defaults** (Recommended)
```csharp
// Change registration
builder.Services.AddMauiML(); // Instead of AddSingleton<IMLInfer, OnnxRuntimeInfer>()

// Code stays the same
public MyService(IMLInfer mlService) { }
```

**Phase 3: Platform-Specific Optimization** (Optional)
```csharp
#if IOS || MACCATALYST
// Use pure CoreML for best performance
builder.Services.AddMauiML(MLBackend.CoreML);
// Convert your .onnx models to .mlmodel
#endif
```

## Summary

This architecture provides:

1. **Flexibility**: Choose backends based on your needs
2. **Performance**: Platform-native acceleration when available
3. **Compatibility**: ONNX fallback ensures it works everywhere
4. **Simplicity**: Unified interface regardless of backend
5. **Future-Proof**: Easy to add new backends without breaking changes

**Recommendation**: Start with `MLPlugin.Default` or `AddMauiML()` for automatic optimization. Only use specific backends when you have platform-specific models or performance requirements.
