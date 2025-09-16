import ctypes
import os
import time
import sys
import logging
import traceback

# Set up detailed logging
logging.basicConfig(
    level=logging.DEBUG,
    format='%(asctime)s [%(levelname)s] [%(name)s] %(message)s',
    handlers=[
        logging.StreamHandler(),
        logging.FileHandler('fingerprint_scanner.log')
    ]
)
logger = logging.getLogger(__name__)

def log_error_with_traceback(e: Exception, prefix: str = "Error"):
    """Helper function to log errors with full traceback"""
    logger.error(f"{prefix}: {str(e)}")
    logger.error("Traceback:", exc_info=True)
    tb_str = ''.join(traceback.format_tb(e.__traceback__))
    logger.error(f"Detailed traceback:\n{tb_str}")

# Try to import Pillow. If not found, print a message but allow the script to run without image display.
try:
    from PIL import Image
    import numpy as np
    PIL_AVAILABLE = True
except ImportError:
    print("\nWarning: Pillow library not found. Install with 'pip install Pillow' to view/save images as PNG.")
    PIL_AVAILABLE = False

# --- Configuration ---
DP_SDK_ROOT = r"C:\Program Files\DigitalPersona\U.are.U SDK"

# Common subdirectories where DLLs might be found
SDK_DLL_DIRS = [
    os.path.join(DP_SDK_ROOT, r"Bin"),
    os.path.join(DP_SDK_ROOT, r"Lib"),
    os.path.join(DP_SDK_ROOT, r"Redist"),
    os.path.join(os.environ.get('WINDIR', ''), 'System32'),
]

# Function to find DLL in common paths
def find_dll(dll_name):
    for sdk_dir in SDK_DLL_DIRS:
        dll_path = os.path.join(sdk_dir, dll_name)
        if os.path.exists(dll_path):
            print(f"Found {dll_name} at: {dll_path}")
            return dll_path
    print(f"Warning: {dll_name} not found in configured SDK paths. Attempting to load by name only (relies on system PATH).")
    return dll_name

DPFPDD_DLL_PATH = find_dll("dpfpdd.dll")
DPFJ_DLL_PATH = find_dll("dpfj.dll")

# --- ctypes Definitions for dpfpdd.dll ---

try:
    dpfpdd = ctypes.WinDLL(DPFPDD_DLL_PATH)
except OSError as e:
    print(f"Failed to load {DPFPDD_DLL_PATH}: {e}")
    print("Ensure the SDK is installed and its DLLs are accessible or in system PATH.")
    print(f"Attempted DPFPDD_DLL_PATH: {DPFPDD_DLL_PATH}")
    sys.exit(1)

# Define common types
DPFPDD_STATUS = ctypes.c_int
DPFPDD_DEV = ctypes.c_void_p

_DP_FACILITY = 0x05BA
def DPERROR(err_code):
    return (ctypes.c_int(err_code).value | (_DP_FACILITY << 16))

DPFPDD_SUCCESS = 0
DPFPDD_E_NOT_IMPLEMENTED = DPERROR(0x0a)
DPFPDD_E_FAILURE = DPERROR(0x0b)
DPFPDD_E_NO_DATA = DPERROR(0x0c)
DPFPDD_E_MORE_DATA = DPERROR(0x0d)
DPFPDD_E_INVALID_PARAMETER = DPERROR(0x14)
DPFPDD_E_INVALID_DEVICE = DPERROR(0x15)
DPFPDD_E_DEVICE_BUSY = DPERROR(0x1e)
DPFPDD_E_DEVICE_FAILURE = DPERROR(0x1f)
DPFPDD_E_PAD_LIBRARY = DPERROR(0x21)
DPFPDD_E_PAD_DATA = DPERROR(0x22)
DPFPDD_E_PAD_LICENSE = DPERROR(0x23)
DPFPDD_E_PAD_FAILURE = DPERROR(0x24)


DPFPDD_IMG_FMT_PIXEL_BUFFER = 0
DPFPDD_IMG_PROC_DEFAULT = 0

DPFPDD_TIMEOUT_INFINITE = 0xFFFFFFFF

# --- SDK Structures ---

MAX_STR_LENGTH = 128
MAX_DEVICE_NAME_LENGTH = 1024

class DPFPDD_VER_INFO(ctypes.Structure):
    _fields_ = [
        ("major", ctypes.c_int),
        ("minor", ctypes.c_int),
        ("maintenance", ctypes.c_int),
    ]

class DPFPDD_HW_DESCR(ctypes.Structure):
    _fields_ = [
        ("vendor_name", ctypes.c_char * MAX_STR_LENGTH),
        ("product_name", ctypes.c_char * MAX_STR_LENGTH),
        ("serial_num", ctypes.c_char * MAX_STR_LENGTH),
    ]

class DPFPDD_HW_ID(ctypes.Structure):
    _fields_ = [
        ("vendor_id", ctypes.c_ushort),
        ("product_id", ctypes.c_ushort),
    ]

class DPFPDD_HW_VERSION(ctypes.Structure):
    _fields_ = [
        ("hw_ver", DPFPDD_VER_INFO),
        ("fw_ver", DPFPDD_VER_INFO),
        ("bcd_rev", ctypes.c_ushort),
    ]

class DPFPDD_DEV_INFO(ctypes.Structure):
    _fields_ = [
        ("size", ctypes.c_uint),
        ("name", ctypes.c_char * MAX_DEVICE_NAME_LENGTH),
        ("descr", DPFPDD_HW_DESCR),
        ("id", DPFPDD_HW_ID),
        ("ver", DPFPDD_HW_VERSION), # Corrected back to DPFPDD_HW_VERSION
        ("modality", ctypes.c_uint),
        ("technology", ctypes.c_uint),
    ]

class DPFPDD_CAPTURE_PARAM(ctypes.Structure):
    _fields_ = [
        ("size", ctypes.c_uint),
        ("image_fmt", ctypes.c_uint),
        ("image_proc", ctypes.c_uint),
        ("image_res", ctypes.c_uint),
    ]

class DPFPDD_IMAGE_INFO(ctypes.Structure):
    _fields_ = [
        ("size", ctypes.c_uint),
        ("width", ctypes.c_uint),
        ("height", ctypes.c_uint),
        ("res", ctypes.c_uint),
        ("bpp", ctypes.c_uint),
    ]

class DPFPDD_CAPTURE_RESULT(ctypes.Structure):
    _fields_ = [
        ("size", ctypes.c_uint),
        ("success", ctypes.c_int),
        ("quality", ctypes.c_uint),
        ("score", ctypes.c_uint),
        ("info", DPFPDD_IMAGE_INFO),
    ]

class DPFPDD_DEV_CAPS_MIN(ctypes.Structure):
    _fields_ = [
        ("size", ctypes.c_uint),
        ("can_capture_image", ctypes.c_int),
        ("can_stream_image", ctypes.c_int),
        ("can_extract_features", ctypes.c_int),
        ("can_match", ctypes.c_int),
        ("can_identify", ctypes.c_int),
        ("has_fp_storage", ctypes.c_int),
        ("indicator_type", ctypes.c_uint),
        ("has_pwr_mgmt", ctypes.c_int),
        ("has_calibration", ctypes.c_int),
        ("piv_compliant", ctypes.c_int),
        ("resolution_cnt", ctypes.c_uint),
        ("resolutions", ctypes.c_uint * 1),
    ]

# --- Define function prototypes for dpfpdd.dll ---

dpfpdd.dpfpdd_init.argtypes = []
dpfpdd.dpfpdd_init.restype = DPFPDD_STATUS

dpfpdd.dpfpdd_exit.argtypes = []
dpfpdd.dpfpdd_exit.restype = DPFPDD_STATUS

dpfpdd.dpfpdd_query_devices.argtypes = [
    ctypes.POINTER(ctypes.c_uint),
    ctypes.POINTER(DPFPDD_DEV_INFO)
]
dpfpdd.dpfpdd_query_devices.restype = DPFPDD_STATUS

dpfpdd.dpfpdd_open.argtypes = [
    ctypes.c_char_p,
    ctypes.POINTER(DPFPDD_DEV)
]
dpfpdd.dpfpdd_open.restype = DPFPDD_STATUS

dpfpdd.dpfpdd_close.argtypes = [DPFPDD_DEV]
dpfpdd.dpfpdd_close.restype = DPFPDD_STATUS

dpfpdd.dpfpdd_get_device_capabilities.argtypes = [
    DPFPDD_DEV,
    ctypes.POINTER(DPFPDD_DEV_CAPS_MIN)
]
dpfpdd.dpfpdd_get_device_capabilities.restype = DPFPDD_STATUS


# Function to get buffer size (not needed anymore since we use fixed size)
dpfpdd.dpfpdd_capture.argtypes = [
    DPFPDD_DEV,
    ctypes.POINTER(DPFPDD_CAPTURE_PARAM),
    ctypes.c_uint,
    ctypes.POINTER(DPFPDD_CAPTURE_RESULT),
    ctypes.POINTER(ctypes.c_uint),
    ctypes.POINTER(ctypes.c_ubyte)
]
dpfpdd.dpfpdd_capture.restype = DPFPDD_STATUS

dpfpdd.dpfpdd_cancel.argtypes = [DPFPDD_DEV]
dpfpdd.dpfpdd_cancel.restype = DPFPDD_STATUS


# --- Helper Functions ---

def check_status(status, func_name):
    """Helper to check SDK function return status."""
    if status == DPFPDD_SUCCESS:
        return True
    else:
        error_messages = {
            DPFPDD_E_NOT_IMPLEMENTED: "API call not implemented",
            DPFPDD_E_FAILURE: "Unspecified failure",
            DPFPDD_E_NO_DATA: "No data available",
            DPFPDD_E_MORE_DATA: "Memory buffer too small (more data needed)",
            DPFPDD_E_INVALID_PARAMETER: "One or more parameters are invalid",
            DPFPDD_E_INVALID_DEVICE: "Reader handle is not valid",
            DPFPDD_E_DEVICE_BUSY: "Another operation is in progress",
            DPFPDD_E_DEVICE_FAILURE: "Reader not working properly / Failed to open/close/start",
            DPFPDD_E_PAD_LIBRARY: "Spoof detection library not found or can't be loaded",
            DPFPDD_E_PAD_DATA: "Spoof detection database/classifier not found or can't be loaded",
            DPFPDD_E_PAD_LICENSE: "Spoof detection license not found or invalid",
            DPFPDD_E_PAD_FAILURE: "Failure to perform spoof detection",
        }
        msg = error_messages.get(status, "Unknown error code")
        print(f"Error in {func_name}: Status = {hex(status)} ({msg})")
        return False

def list_devices():
    """Lists available DigitalPersona fingerprint devices."""
    MAX_DEVICES = 10
    
    while True:
        # Create an array of DPFPDD_DEV_INFO structures
        dev_infos_array = (DPFPDD_DEV_INFO * MAX_DEVICES)()
        for i in range(MAX_DEVICES):
            dev_infos_array[i].size = ctypes.sizeof(DPFPDD_DEV_INFO)

        count = ctypes.c_uint(MAX_DEVICES)
        
        # Call the query devices function
        status = dpfpdd.dpfpdd_query_devices(ctypes.byref(count), dev_infos_array)
        
        if status == DPFPDD_SUCCESS:
            devices = []
            for i in range(count.value):
                dev_info = dev_infos_array[i]
                devices.append({
                    "name": dev_info.name.decode('utf-8'),
                    "vendor_name": dev_info.descr.vendor_name.decode('utf-8'),
                    "product_name": dev_info.descr.product_name.decode('utf-8'),
                    "serial_num": dev_info.descr.serial_num.decode('utf-8'),
                    "vendor_id": dev_info.id.vendor_id,
                    "product_id": dev_info.id.product_id,
                })
            return devices
        elif status == DPFPDD_E_MORE_DATA:
            print(f"More devices found ({count.value}) than allocated space ({MAX_DEVICES}). Resizing buffer.")
            MAX_DEVICES = count.value
            if MAX_DEVICES == 0:
                print("Warning: SDK returned 0 devices after DPFPDD_E_MORE_DATA. Exiting device query loop.")
                return []
            continue
        else:
            check_status(status, "dpfpdd_query_devices")
            return []


def capture_fingerprint_image(device_name=None):
    """
    Captures a single fingerprint image using dpfpdd_capture.
    If device_name is None, it will try to use the U.are.U device.
    Returns (image_data_bytes, image_info_dict, quality_flags) or (None, None, None) on failure.
    """
    dev = DPFPDD_DEV(None)
    
    status = dpfpdd.dpfpdd_init()
    if not check_status(status, "dpfpdd_init"):
        return None, None, None

    try:
        if device_name is None:
            devices = list_devices()
            if not devices:
                print("No DigitalPersona fingerprint devices found.")
                return None, None, None
            
            print("Found devices:")
            for i, d in enumerate(devices):
                print(f"  {i+1}. Name: '{d['name']}', Product: '{d['product_name']}', Serial: '{d['serial_num']}'")

            # Select U.are.U device
            uareu_device = next(
                (d for d in devices if "U.are.U" in d["product_name"]),
                None
            )
            
            if not uareu_device:
                print("No U.are.U device found!")
                return None, None, None

            device_name = uareu_device["name"]
            print(f"Using U.are.U device: '{device_name}'")

        status = dpfpdd.dpfpdd_open(device_name.encode('utf-8'), ctypes.byref(dev))
        if not check_status(status, "dpfpdd_open"):
            return None, None, None

        print("Device opened. Place your finger on the scanner...")

        dev_caps_obj = None

        dev_caps_min = DPFPDD_DEV_CAPS_MIN()
        dev_caps_min.size = ctypes.sizeof(DPFPDD_DEV_CAPS_MIN)

        status = dpfpdd.dpfpdd_get_device_capabilities(dev, ctypes.byref(dev_caps_min))

        if status == DPFPDD_E_MORE_DATA:
            required_caps_size = dev_caps_min.size
            print(f"Device capabilities struct too small. Re-allocating for {required_caps_size} bytes.")

            base_caps_size = ctypes.sizeof(DPFPDD_DEV_CAPS_MIN) - ctypes.sizeof(ctypes.c_uint)
            num_additional_resolutions = (required_caps_size - base_caps_size) // ctypes.sizeof(ctypes.c_uint)
            
            class DPFPDD_DEV_CAPS_FULL(ctypes.Structure):
                _fields_ = [
                    ("size", ctypes.c_uint),
                    ("can_capture_image", ctypes.c_int),
                    ("can_stream_image", ctypes.c_int),
                    ("can_extract_features", ctypes.c_int),
                    ("can_match", ctypes.c_int),
                    ("can_identify", ctypes.c_int),
                    ("has_fp_storage", ctypes.c_int),
                    ("indicator_type", ctypes.c_uint),
                    ("has_pwr_mgmt", ctypes.c_int),
                    ("has_calibration", ctypes.c_int),
                    ("piv_compliant", ctypes.c_int),
                    ("resolution_cnt", ctypes.c_uint),
                    ("resolutions", ctypes.c_uint * (num_additional_resolutions + 1)),
                ]

            dev_caps_obj = DPFPDD_DEV_CAPS_FULL()
            dev_caps_obj.size = required_caps_size
            status = dpfpdd.dpfpdd_get_device_capabilities(dev, ctypes.byref(dev_caps_obj))
            
        elif status == DPFPDD_SUCCESS:
            dev_caps_obj = dev_caps_min
        
        if not check_status(status, "dpfpdd_get_device_capabilities"):
            print("Failed to get device capabilities. Cannot determine supported resolutions.")
            return None, None, None

        preferred_resolution = 0
        if dev_caps_obj and dev_caps_obj.resolution_cnt > 0:
            preferred_resolution = dev_caps_obj.resolutions[0]
            print(f"Device supports {dev_caps_obj.resolution_cnt} resolutions. Using: {preferred_resolution} DPI.")
        else:
            print("Warning: Device reported no supported resolutions or capabilities not retrieved. Using default (0).")


        capture_parm = DPFPDD_CAPTURE_PARAM()
        capture_parm.size = ctypes.sizeof(DPFPDD_CAPTURE_PARAM)
        capture_parm.image_fmt = DPFPDD_IMG_FMT_PIXEL_BUFFER
        capture_parm.image_proc = DPFPDD_IMG_PROC_DEFAULT
        capture_parm.image_res = preferred_resolution

        capture_result = DPFPDD_CAPTURE_RESULT()
        capture_result.size = ctypes.sizeof(DPFPDD_CAPTURE_RESULT)

        # We will allocate a buffer large enough for a common image, 
        # but the SDK will tell us the exact required size.
        # Max expected for 500 DPI, 500x500 is 250000 bytes.
        INITIAL_IMAGE_BUFFER_SIZE = 500 * 500
        
        image_buffer_ptr = ctypes.POINTER(ctypes.c_ubyte)()
        image_buffer = ctypes.create_string_buffer(INITIAL_IMAGE_BUFFER_SIZE)
        image_buffer_ptr = ctypes.cast(image_buffer, ctypes.POINTER(ctypes.c_ubyte))
        actual_image_size = ctypes.c_uint(INITIAL_IMAGE_BUFFER_SIZE)

        while True:
            status = dpfpdd.dpfpdd_capture(
                dev,
                ctypes.byref(capture_parm),
                DPFPDD_TIMEOUT_INFINITE,
                ctypes.byref(capture_result),
                ctypes.byref(actual_image_size),
                image_buffer_ptr
            )

            if status == DPFPDD_SUCCESS:
                break
            elif status == DPFPDD_E_MORE_DATA:
                required_size = actual_image_size.value
                print(f"Image buffer too small. Re-allocating for {required_size} bytes.")
                image_buffer = ctypes.create_string_buffer(required_size)
                image_buffer_ptr = ctypes.cast(image_buffer, ctypes.POINTER(ctypes.c_ubyte))
                actual_image_size.value = required_size
            else:
                check_status(status, "dpfpdd_capture")
                return None, None, None

        if capture_result.success == 0:
            print(f"Capture operation reported failure by SDK. Quality flags: {hex(capture_result.quality)}")
            return None, None, None

        print(f"Fingerprint captured! Score: {capture_result.score}, Quality Flags: {hex(capture_result.quality)}")
        print(f"Image Info: Width={capture_result.info.width}, Height={capture_result.info.height}, Res={capture_result.info.res}, BPP={capture_result.info.bpp}")
        
        # *** Crucial Change Here: Only take the expected pixel data size ***
        # The SDK returns `actual_image_size.value` as the size of the buffer used/required.
        # However, for a raw PIXEL_BUFFER image, the actual pixel data should be width * height * (bpp/8).
        # We need to explicitly slice the buffer to this expected size.
        expected_pixel_data_size = capture_result.info.width * capture_result.info.height * (capture_result.info.bpp // 8)
        
        if actual_image_size.value < expected_pixel_data_size:
             print(f"Warning: SDK reported image size {actual_image_size.value} but expected pixel data is {expected_pixel_data_size}. This might indicate a problem.")
             # We will still proceed with the expected_pixel_data_size for reshaping.
             # If actual_image_size.value is significantly smaller, it could mean data truncation.
             # For this U.are.U device, it's usually the other way around (SDK reports larger buffer size)

        # Use the expected pixel data size for slicing, not actual_image_size.value directly
        image_data_bytes = bytes(image_buffer[:expected_pixel_data_size])

        image_info_dict = {
            "width": capture_result.info.width,
            "height": capture_result.info.height,
            "resolution": capture_result.info.res,
            "bpp": capture_result.info.bpp,
        }
        
        return image_data_bytes, image_info_dict, capture_result.quality

    finally:
        if dev and dev.value:
            dpfpdd.dpfpdd_close(dev)
        dpfpdd.dpfpdd_exit()

# --- Main Execution ---
if __name__ == "__main__":
    print("Attempting to capture fingerprint...")
    image_bytes, img_info, quality_flags = capture_fingerprint_image()

    if image_bytes:
        print(f"Captured {len(image_bytes)} bytes of pixel data.") # Changed message
        if img_info:
            raw_filename = f"captured_fingerprint_raw_{img_info['width']}x{img_info['height']}.raw"
            try:
                with open(raw_filename, "wb") as f:
                    f.write(image_bytes)
                print(f"Raw fingerprint image saved to {raw_filename}")
            except Exception as e:
                print(f"Could not save raw image: {e}")

            if PIL_AVAILABLE:
                try:
                    width = img_info['width']
                    height = img_info['height']
                    bpp = img_info['bpp']

                    if bpp == 8:
                        # The numpy reshape expects (height, width) for image data
                        # Ensure we use the correct dimensions from img_info
                        np_image = np.frombuffer(image_bytes, dtype=np.uint8).reshape((height, width))
                        img = Image.fromarray(np_image, 'L')
                        
                        output_png_filename = f"captured_fingerprint_{width}x{height}.png"
                        img.save(output_png_filename)
                        print(f"PNG image saved to {output_png_filename}")
                        
                        # Don't show image when running as API (server environment)
                        # img.show()
                        
                    else:
                        print(f"Cannot automatically display/convert image with {bpp} BPP. Only 8-bit grayscale is supported for automatic display.")
                except Exception as e:
                    print(f"Error during image conversion/display with Pillow: {e}")
            else:
                print("\nPillow is not installed. To view the image, use a raw image viewer and specify:")
                print(f"  Width={img_info['width']}, Height={img_info['height']}, BitsPerPixel={img_info['bpp']}")
    else:
        print("Fingerprint capture failed.")



import win32com.client
import pythoncom
import time
import numpy as np
from PIL import Image
import io
import logging
import os

logging.basicConfig(level=logging.DEBUG)
logger = logging.getLogger(__name__)

class FingerprintScanner:
    def __init__(self):
        self.device = None
        self._initialize_reader()

    def _initialize_reader(self):
        try:
            logger.debug("Starting fingerprint reader initialization...")
            
            # First, query the number of devices
            num_devices = ctypes.c_uint(0)
            logger.debug("Querying for connected devices...")
            
            # First call to get device count
            status = dpfpdd.dpfpdd_query_devices(ctypes.byref(num_devices), None)
            
            logger.debug(f"Query status: 0x{status:x}, Number of devices: {num_devices.value}")
            
            if status != DPFPDD_SUCCESS and status != DPFPDD_E_MORE_DATA:
                error_msg = f"Failed to query devices: Status = 0x{status:x}"
                logger.error(error_msg)
                if status == DPFPDD_E_FAILURE:
                    error_msg += " (Unspecified failure - check device permissions)"
                elif status == DPFPDD_E_INVALID_PARAMETER:
                    error_msg += " (Invalid parameter - internal error)"
                raise Exception(error_msg)
            
            if num_devices.value == 0:
                error_msg = "No fingerprint readers found. Please check device connection."
                logger.error(error_msg)
                raise Exception(error_msg)
            
            logger.info(f"Found {num_devices.value} fingerprint device(s)")
            
            # Create array of device info structures
            devices = (DPFPDD_DEV_INFO * num_devices.value)()
            
            # Initialize each structure
            for dev in devices:
                dev.size = ctypes.sizeof(DPFPDD_DEV_INFO)
                dev.descr.vendor_name = b"\0" * MAX_STR_LENGTH
                dev.descr.product_name = b"\0" * MAX_STR_LENGTH
                dev.descr.serial_num = b"\0" * MAX_STR_LENGTH
                dev.name = b"\0" * MAX_DEVICE_NAME_LENGTH
            
            # Query devices again to get their info
            status = dpfpdd.dpfpdd_query_devices(ctypes.byref(num_devices), devices)
            
            if status != DPFPDD_SUCCESS:
                raise Exception(f"Failed to get device info: Status = 0x{status:x}")
            
            # Find U.are.U device
            uareu_device = None
            for dev in devices:
                logger.debug(f"Found device: {dev.descr.product_name.decode('utf-8', errors='ignore')}")
                if b"U.are.U" in dev.descr.product_name:
                    uareu_device = dev
                    break
            
            if not uareu_device:
                raise Exception("U.are.U scanner not found")
            
            logger.debug(f"Using device: {uareu_device.descr.product_name.decode('utf-8', errors='ignore')}")
            
            # Open the device
            device = DPFPDD_DEV()
            status = dpfpdd.dpfpdd_open(uareu_device.name, ctypes.byref(device))
            
            if status != DPFPDD_SUCCESS:
                raise Exception(f"Failed to open device: Status = 0x{status:x}")
            
            self.device = device
            logger.info("Scanner initialized successfully")

        except Exception as e:
            logger.error(f"Scanner initialization failed: {str(e)}")
            raise

    def capture_fingerprint(self):
        if not self.device:
            error_msg = "Scanner not initialized. Please check device connection."
            logger.error(error_msg)
            raise Exception(error_msg)

        try:
            logger.info("Starting fingerprint capture...")
            
            # Get capabilities - first call to get required size
            caps = DPFPDD_DEV_CAPS_MIN()
            caps.size = ctypes.sizeof(DPFPDD_DEV_CAPS_MIN)
            status = dpfpdd.dpfpdd_get_device_capabilities(self.device, ctypes.byref(caps))
            
            if status == DPFPDD_E_MORE_DATA:
                # We need a larger structure with more resolution entries
                resolution_count = caps.resolution_cnt
                logger.debug(f"Device has {resolution_count} resolutions, adjusting structure size")
                
                # Create a new structure type with the correct number of resolution entries
                class DPFPDD_DEV_CAPS(ctypes.Structure):
                    _fields_ = [
                        ("size", ctypes.c_uint),
                        ("can_capture_image", ctypes.c_int),
                        ("can_stream_image", ctypes.c_int),
                        ("can_extract_features", ctypes.c_int),
                        ("can_match", ctypes.c_int),
                        ("can_identify", ctypes.c_int),
                        ("has_fp_storage", ctypes.c_int),
                        ("indicator_type", ctypes.c_uint),
                        ("has_pwr_mgmt", ctypes.c_int),
                        ("has_calibration", ctypes.c_int),
                        ("piv_compliant", ctypes.c_int),
                        ("resolution_cnt", ctypes.c_uint),
                        ("resolutions", ctypes.c_uint * resolution_count),
                    ]
                
                # Create new capabilities structure with correct size
                caps = DPFPDD_DEV_CAPS()
                caps.size = ctypes.sizeof(DPFPDD_DEV_CAPS)
                status = dpfpdd.dpfpdd_get_device_capabilities(self.device, ctypes.byref(caps))
            
            if status != DPFPDD_SUCCESS:
                raise Exception(f"Failed to get device capabilities: Status = 0x{status:x}")
            
            # Log available resolutions
            try:
                res_list = list(caps.resolutions[:caps.resolution_cnt])
                logger.debug(f"Available device resolutions: {res_list}")
            except Exception as e:
                logger.error(f"Error reading device resolutions: {e}")
                res_list = [500]  # fallback

            # Use the first available resolution
            if res_list:
                chosen_res = res_list[0]
            else:
                chosen_res = 500  # fallback
            logger.debug(f"Using image resolution: {chosen_res}")

            # Use dynamic buffer allocation like the working reference script
            capture_params = DPFPDD_CAPTURE_PARAM()
            capture_params.size = ctypes.sizeof(DPFPDD_CAPTURE_PARAM)
            capture_params.image_fmt = DPFPDD_IMG_FMT_PIXEL_BUFFER
            capture_params.image_proc = DPFPDD_IMG_PROC_DEFAULT
            capture_params.image_res = chosen_res

            capture_result = DPFPDD_CAPTURE_RESULT()
            capture_result.size = ctypes.sizeof(DPFPDD_CAPTURE_RESULT)

            # Start with a buffer large enough for common images
            INITIAL_IMAGE_BUFFER_SIZE = 500 * 500
            image_buffer = ctypes.create_string_buffer(INITIAL_IMAGE_BUFFER_SIZE)
            image_buffer_ptr = ctypes.cast(image_buffer, ctypes.POINTER(ctypes.c_ubyte))
            actual_image_size = ctypes.c_uint(INITIAL_IMAGE_BUFFER_SIZE)

            logger.debug(f"Capture params: size={capture_params.size}, fmt={capture_params.image_fmt}, proc={capture_params.image_proc}, res={capture_params.image_res}")
            logger.debug(f"Capture result struct size: {capture_result.size}")
            logger.debug(f"Initial image buffer size: {actual_image_size.value}")

            while True:
                status = dpfpdd.dpfpdd_capture(
                    self.device,
                    ctypes.byref(capture_params),
                    DPFPDD_TIMEOUT_INFINITE,
                    ctypes.byref(capture_result),
                    ctypes.byref(actual_image_size),
                    image_buffer_ptr
                )

                if status == DPFPDD_SUCCESS:
                    break
                elif status == DPFPDD_E_MORE_DATA:
                    required_size = actual_image_size.value
                    logger.debug(f"Image buffer too small. Re-allocating for {required_size} bytes.")
                    image_buffer = ctypes.create_string_buffer(required_size)
                    image_buffer_ptr = ctypes.cast(image_buffer, ctypes.POINTER(ctypes.c_ubyte))
                    actual_image_size.value = required_size
                else:
                    raise Exception(f"Failed to capture: Status = 0x{status:x}")

            if capture_result.success == 0:
                logger.error(f"Capture operation reported failure by SDK. Quality flags: {hex(capture_result.quality)}")
                raise Exception("Capture operation failed")
                
            logger.info(f"Fingerprint captured! Score: {capture_result.score}, Quality Flags: {hex(capture_result.quality)}")
            logger.info(f"Image Info: Width={capture_result.info.width}, Height={capture_result.info.height}, Res={capture_result.info.res}, BPP={capture_result.info.bpp}")
            
            # Get actual image dimensions from capture result
            width = capture_result.info.width
            height = capture_result.info.height
            bpp = capture_result.info.bpp
            
            # Calculate expected pixel data size
            expected_pixel_data_size = width * height * (bpp // 8)
            
            if actual_image_size.value < expected_pixel_data_size:
                logger.warning(f"SDK reported image size {actual_image_size.value} but expected pixel data is {expected_pixel_data_size}.")
            
            # Use the expected pixel data size for slicing
            image_data_bytes = bytes(image_buffer[:expected_pixel_data_size])

            # Convert to PIL Image
            if PIL_AVAILABLE:
                try:
                    import numpy as np
                    # Reshape the raw bytes into a 2D array
                    np_image = np.frombuffer(image_data_bytes, dtype=np.uint8).reshape((height, width))
                    img = Image.fromarray(np_image, 'L')
                    
                    # Save to bytes
                    import io
                    img_byte_arr = io.BytesIO()
                    img.save(img_byte_arr, format='PNG')
                    logger.info("Fingerprint captured and converted to PNG successfully")
                    return img_byte_arr.getvalue()
                except Exception as e:
                    logger.error(f"Error converting image: {str(e)}")
                    raise
            else:
                # If PIL is not available, return raw bytes
                logger.info("PIL not available, returning raw image data")
                return image_data_bytes

        except Exception as e:
            logger.error(f"Capture error: {str(e)}")
            raise
            logger.error(f"Capture error: {str(e)}")
            raise
        finally:
            try:
                self.reader.StopCapture()
            except:
                pass

    def close(self):
        if self.device:
            try:
                dpfpdd.dpfpdd_close(self.device)
            except:
                pass
            self.device = None

    def __del__(self):
        self.close()

# Test function
def test_scanner():
    try:
        print("\nTesting fingerprint scanner...")
        devices = list_devices()
        
        # Find U.are.U device
        uareu_device = next(
            (d for d in devices if "U.are.U" in d["product_name"]),
            None
        )
        
        if not uareu_device:
            print("Error: U.are.U scanner not found!")
            return
            
        print(f"Found U.are.U device: {uareu_device['product_name']}")
        
        # Capture using specific device
        image_bytes, img_info, quality = capture_fingerprint_image(uareu_device["name"])
        
        if image_bytes:
            print(f"Captured {len(image_bytes)} bytes of pixel data.")
            if img_info:
                png_filename = f"fingerprint_{img_info['width']}x{img_info['height']}.png"
                with open(png_filename, "wb") as f:
                    f.write(image_bytes)
                print(f"Image saved as {png_filename}")
        else:
            print("Capture failed!")
            
    except Exception as e:
        print(f"Test failed: {str(e)}")

if __name__ == "__main__":
    test_scanner()