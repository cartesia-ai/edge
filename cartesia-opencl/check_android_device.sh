#!/bin/bash

ADB_PATH="$HOME/Library/Android/sdk/platform-tools/adb"

echo "=========================================="
echo "Android Device Hardware & OpenCL Check"
echo "=========================================="
echo ""

# Check if device is connected
echo "1. Checking device connection..."
$ADB_PATH devices -l
echo ""

# Device basic info
echo "2. Device Information:"
echo "   Model: $($ADB_PATH shell getprop ro.product.model)"
echo "   Brand: $($ADB_PATH shell getprop ro.product.brand)"
echo "   Manufacturer: $($ADB_PATH shell getprop ro.product.manufacturer)"
echo "   Device: $($ADB_PATH shell getprop ro.product.device)"
echo "   Hardware: $($ADB_PATH shell getprop ro.hardware)"
echo "   Board Platform: $($ADB_PATH shell getprop ro.board.platform)"
echo "   Chipset: $($ADB_PATH shell getprop ro.chipname)"
echo "   Android Version: $($ADB_PATH shell getprop ro.build.version.release)"
echo "   SDK Version: $($ADB_PATH shell getprop ro.build.version.sdk)"
echo ""

# CPU Info
echo "3. CPU Information:"
$ADB_PATH shell "cat /proc/cpuinfo | grep -E 'processor|Hardware|model name' | head -10"
echo ""

# GPU Info
echo "4. GPU Information:"
echo "   GPU Renderer: $($ADB_PATH shell getprop ro.opengles.version)"
$ADB_PATH shell "dumpsys SurfaceFlinger | grep -i 'GLES\|gpu\|renderer' | head -5"
echo ""

# Check for OpenCL libraries
echo "5. Checking for OpenCL libraries:"
$ADB_PATH shell "find /system/lib* /vendor/lib* -name '*opencl*' -o -name '*OpenCL*' 2>/dev/null | head -10"
echo ""

# Check for OpenCL devices via CLInfo if available
echo "6. Checking OpenCL support:"
if $ADB_PATH shell "which clinfo" > /dev/null 2>&1; then
    echo "   clinfo found, running..."
    $ADB_PATH shell clinfo
else
    echo "   clinfo not available on device"
    echo "   Checking for OpenCL files in /vendor/lib* and /system/lib*:"
    $ADB_PATH shell "ls -la /vendor/lib*/libOpenCL.so /system/lib*/libOpenCL.so 2>/dev/null"
fi
echo ""

# Adreno GPU info (common on Qualcomm devices)
echo "7. Checking for Adreno GPU (Qualcomm):"
$ADB_PATH shell "cat /sys/class/kgsl/kgsl-3d0/gpu_model 2>/dev/null || echo 'Not available'"
echo ""

# Mali GPU info (common on MediaTek/Exynos devices)
echo "8. Checking for Mali GPU:"
$ADB_PATH shell "ls -la /sys/class/misc/mali*/version 2>/dev/null || echo 'Not available'"
echo ""

# Vulkan support (often indicates compute capability)
echo "9. Checking Vulkan support:"
$ADB_PATH shell "ls -la /vendor/lib*/libvulkan.so /system/lib*/libvulkan.so 2>/dev/null || echo 'Not available'"
echo ""

# All OpenCL related files
echo "10. All OpenCL related files:"
$ADB_PATH shell "find /system /vendor -type f -iname '*opencl*' 2>/dev/null"
echo ""

echo "=========================================="
echo "Done!"
echo "=========================================="

