import platform
import psutil
import subprocess

os_info = f"{platform.system()} {platform.release()}"
python_version = platform.python_version()

ram_gb = round(psutil.virtual_memory().total / (1024**3))

cpu_model = platform.processor()
physical_cores = psutil.cpu_count(logical=False)
total_threads = psutil.cpu_count(logical=True)

gpu_info = "None or Not Detected"
try:
    smi_output = subprocess.check_output(
        ["nvidia-smi", "--query-gpu=name,memory.total", "--format=csv,noheader"],
        encoding="utf-8",
    ).strip()
    gpu_info = smi_output.replace("\n", " | ")
except (subprocess.FileNotFoundError, Exception):
    gpu_info = "No NVIDIA GPU detected (or nvidia-smi not in PATH)"

print(f"""
### Hardware & Environment Specifications
* **OS:** {os_info}
* **Python Version:** {python_version}
* **CPU:** {cpu_model} ({physical_cores} Cores / {total_threads} Threads)
* **RAM:** {ram_gb} GB
* **GPU:** {gpu_info}
""")
