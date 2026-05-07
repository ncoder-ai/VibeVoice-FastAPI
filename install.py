#!/usr/bin/env python3
"""
VibeVoice-FastAPI Interactive Installer

Cross-platform setup wizard that detects your system, asks configuration
questions, generates .env, and runs the appropriate setup steps.

Usage: python install.py
"""

import os
import platform
import re
import shutil
import subprocess
import sys
from pathlib import Path

SCRIPT_DIR = Path(__file__).resolve().parent

# ============================================================
# Helpers
# ============================================================

def print_header(title):
    print()
    print("=" * 60)
    print(f"  {title}")
    print("=" * 60)
    print()


def print_step(step_num, total, description):
    print(f"\n--- Step {step_num}/{total}: {description} ---\n")


def ask_choice(prompt, options, default=1):
    """Present numbered options and return the selected index (0-based)."""
    print(prompt)
    for i, (label, description) in enumerate(options, 1):
        marker = " (default)" if i == default else ""
        print(f"  {i}) {label}{marker}")
        if description:
            print(f"     {description}")
    while True:
        raw = input(f"\nChoice [{default}]: ").strip()
        if raw == "":
            return default - 1
        try:
            choice = int(raw)
            if 1 <= choice <= len(options):
                return choice - 1
        except ValueError:
            pass
        print(f"  Please enter a number between 1 and {len(options)}.")


def ask_yesno(prompt, default=False):
    """Ask a yes/no question. Returns bool."""
    hint = "y/N" if not default else "Y/n"
    while True:
        raw = input(f"{prompt} [{hint}]: ").strip().lower()
        if raw == "":
            return default
        if raw in ("y", "yes"):
            return True
        if raw in ("n", "no"):
            return False
        print("  Please enter y or n.")


def ask_string(prompt, default=""):
    """Ask for a string value with a default."""
    if default:
        raw = input(f"{prompt} [{default}]: ").strip()
        return raw if raw else default
    else:
        while True:
            raw = input(f"{prompt}: ").strip()
            if raw:
                return raw
            print("  A value is required.")


def ask_path(prompt, default="", must_exist=False):
    """Ask for a filesystem path with optional existence check."""
    while True:
        if default:
            raw = input(f"{prompt} [{default}]: ").strip()
            path = raw if raw else default
        else:
            raw = input(f"{prompt}: ").strip()
            if not raw:
                print("  A path is required.")
                continue
            path = raw
        if must_exist and not os.path.exists(path):
            print(f"  Path not found: {path}")
            if not ask_yesno("  Use it anyway?", default=False):
                continue
        return path


# ============================================================
# Detection
# ============================================================

def detect_os():
    """Return 'linux', 'macos', or 'windows'."""
    system = platform.system().lower()
    if system == "darwin":
        return "macos"
    if system == "windows":
        return "windows"
    return "linux"


def detect_gpus():
    """Run nvidia-smi and return list of (index, name, vram_mb) tuples."""
    gpus = []
    try:
        result = subprocess.run(
            ["nvidia-smi", "--query-gpu=index,name,memory.total",
             "--format=csv,noheader,nounits"],
            capture_output=True, text=True, timeout=10,
        )
        if result.returncode == 0:
            for line in result.stdout.strip().splitlines():
                parts = [p.strip() for p in line.split(",")]
                if len(parts) >= 3:
                    idx = parts[0]
                    name = parts[1]
                    vram = parts[2]
                    gpus.append((idx, name, vram))
    except (FileNotFoundError, subprocess.TimeoutExpired):
        pass
    return gpus


def detect_hf_cache():
    """Return the default HuggingFace cache directory."""
    hf_home = os.environ.get("HF_HOME")
    if hf_home:
        return os.path.join(hf_home, "hub")
    xdg = os.environ.get("XDG_CACHE_HOME")
    if xdg:
        return os.path.join(xdg, "huggingface")
    return os.path.join(str(Path.home()), ".cache", "huggingface")


def check_docker():
    """Check if docker and docker compose are available."""
    try:
        subprocess.run(["docker", "version"], capture_output=True, timeout=10)
        # Try "docker compose" (v2) first, then "docker-compose" (v1)
        r = subprocess.run(["docker", "compose", "version"],
                           capture_output=True, timeout=10)
        if r.returncode == 0:
            return True, ["docker", "compose"]
        r = subprocess.run(["docker-compose", "version"],
                           capture_output=True, timeout=10)
        if r.returncode == 0:
            return True, ["docker-compose"]
        return False, []
    except (FileNotFoundError, subprocess.TimeoutExpired):
        return False, []


# ============================================================
# Config generation
# ============================================================

def generate_env(config, mode):
    """Generate .env file from config dict. Uses the appropriate template."""
    template_name = "docker-env.example" if mode == "docker" else "env.example"
    template_path = SCRIPT_DIR / template_name

    if not template_path.exists():
        print(f"  Warning: {template_name} not found, generating .env from scratch.")
        lines = []
    else:
        lines = template_path.read_text().splitlines()

    # Map of env var name -> value to set
    env_map = {}

    env_map["VIBEVOICE_MODEL_PATH"] = config["model_path"]
    env_map["VIBEVOICE_DEVICE"] = config["device"]
    env_map["API_PORT"] = str(config["port"])

    if config.get("gpu_id") is not None:
        env_map["CUDA_VISIBLE_DEVICES"] = str(config["gpu_id"])

    if config.get("voices_dir"):
        env_map["VOICES_DIR"] = config["voices_dir"]

    if config.get("torch_compile"):
        env_map["TORCH_COMPILE"] = "true"
        env_map["TORCH_COMPILE_MODE"] = "max-autotune"

    if config.get("quantization"):
        env_map["VIBEVOICE_QUANTIZATION"] = config["quantization"]

    if mode == "docker" and config.get("hf_cache_dir"):
        env_map["HF_CACHE_DIR"] = config["hf_cache_dir"]

    # Process template lines: uncomment and set values for keys in env_map
    output_lines = []
    keys_written = set()
    for line in lines:
        stripped = line.lstrip("# ").strip()
        # Check if this line (commented or not) sets a key we want to override
        matched_key = None
        for key in env_map:
            if stripped.startswith(key + "=") or line.strip().startswith(key + "="):
                matched_key = key
                break
        if matched_key and matched_key not in keys_written:
            output_lines.append(f"{matched_key}={env_map[matched_key]}")
            keys_written.add(matched_key)
        else:
            output_lines.append(line)

    # Append any keys that weren't in the template
    for key, value in env_map.items():
        if key not in keys_written:
            output_lines.append(f"{key}={value}")

    env_path = SCRIPT_DIR / ".env"
    env_path.write_text("\n".join(output_lines) + "\n")
    return env_path


def update_docker_gpu(gpu_id):
    """Update docker-compose.yml device_ids to use the selected GPU."""
    compose_path = SCRIPT_DIR / "docker-compose.yml"
    if not compose_path.exists():
        print("  Warning: docker-compose.yml not found, skipping GPU config.")
        return

    content = compose_path.read_text()

    # Backup
    backup_path = SCRIPT_DIR / "docker-compose.yml.bak"
    backup_path.write_text(content)
    print(f"  Backed up docker-compose.yml to docker-compose.yml.bak")

    # Replace device_ids line
    new_content = re.sub(
        r"(device_ids:\s*\[)[^\]]*(\])",
        rf"\g<1>'{gpu_id}'\2",
        content,
    )
    compose_path.write_text(new_content)
    print(f"  Updated docker-compose.yml to use GPU {gpu_id}")


# ============================================================
# Setup runners
# ============================================================

def run_docker_setup(compose_cmd):
    """Build and start Docker containers."""
    print_header("Building and Starting Docker Containers")
    print("This may take a while on first run (downloading model + building image)...\n")

    try:
        subprocess.run(compose_cmd + ["build"], cwd=SCRIPT_DIR, check=True)
        print()
        subprocess.run(compose_cmd + ["up", "-d"], cwd=SCRIPT_DIR, check=True)
        return True
    except subprocess.CalledProcessError as e:
        print(f"\n  Docker command failed with exit code {e.returncode}")
        return False


def run_baremetal_setup(detected_os):
    """Run baremetal setup: setup.sh on Linux/macOS, Python steps on Windows."""
    print_header("Running Baremetal Setup")

    if detected_os in ("linux", "macos"):
        setup_script = SCRIPT_DIR / "setup.sh"
        if setup_script.exists():
            print("Running setup.sh...\n")
            try:
                subprocess.run(["bash", str(setup_script)],
                               cwd=SCRIPT_DIR, check=True)
                return True
            except subprocess.CalledProcessError as e:
                print(f"\n  setup.sh failed with exit code {e.returncode}")
                return False
        else:
            print("  setup.sh not found. Please install dependencies manually.")
            return False
    else:
        # Windows: run equivalent Python steps
        print("Running Windows setup steps...\n")

        # Find Python 3.10-3.12 (prefer 3.12 to match Docker)
        python = None
        for cmd in ["py -3.12", "py -3.11", "py -3.10",
                     "python3.12", "python3.11", "python3.10",
                     "python3", "python"]:
            try:
                parts = cmd.split()
                r = subprocess.run(
                    parts + ["-c", "import sys; print(sys.version_info.minor)"],
                    capture_output=True, text=True, timeout=5,
                )
                if r.returncode == 0:
                    minor = int(r.stdout.strip())
                    if 10 <= minor <= 12:
                        python = parts
                        ver = subprocess.run(
                            parts + ["--version"],
                            capture_output=True, text=True,
                        ).stdout.strip()
                        print(f"  Found compatible Python: {ver}")
                        break
            except (FileNotFoundError, subprocess.TimeoutExpired, ValueError):
                continue

        if python is None:
            print("  ERROR: Python 3.10, 3.11, or 3.12 is required.")
            print("  Download Python 3.12 from: https://www.python.org/downloads/")
            print("  Make sure to check 'Add Python to PATH' during installation.")
            return False

        venv_dir = SCRIPT_DIR / "venv"

        if not venv_dir.exists():
            print("  Creating virtual environment...")
            subprocess.run(python + ["-m", "venv", str(venv_dir)], check=True)

        # Use "python -m pip" instead of "pip" directly to avoid
        # Windows file locking issues when pip tries to upgrade itself
        if detected_os == "windows":
            venv_python = str(venv_dir / "Scripts" / "python.exe")
        else:
            venv_python = str(venv_dir / "bin" / "python")
        venv_pip = [venv_python, "-m", "pip"]

        errors = []

        def run_step(description, cmd, required=True):
            """Run a setup step, log failures, continue if not required."""
            print(f"  {description}...")
            r = subprocess.run(cmd)
            if r.returncode != 0:
                msg = f"{description} failed (exit code {r.returncode})"
                print(f"  WARNING: {msg}")
                if required:
                    errors.append(msg)
                return False
            return True

        run_step("Upgrading pip",
                 venv_pip + ["install", "--upgrade", "pip", "wheel",
                             "setuptools"])

        # Detect CUDA version for correct PyTorch index
        torch_index = "https://download.pytorch.org/whl/cu128"
        try:
            result = subprocess.run(
                ["nvidia-smi", "--query-gpu=driver_version",
                 "--format=csv,noheader"],
                capture_output=True, text=True, timeout=10,
            )
            if result.returncode == 0:
                print("  NVIDIA GPU detected")
            else:
                torch_index = "https://download.pytorch.org/whl/cpu"
                print("  No NVIDIA GPU detected, using CPU PyTorch")
        except (FileNotFoundError, subprocess.TimeoutExpired):
            torch_index = "https://download.pytorch.org/whl/cpu"
            print("  nvidia-smi not found, using CPU PyTorch")

        # Try versioned install first, fall back to latest
        if not run_step("Installing PyTorch 2.8.x",
                        venv_pip + ["install", "torch==2.8.*", "torchaudio",
                                    "--index-url", torch_index],
                        required=False):
            errors.pop() if errors and "PyTorch" in errors[-1] else None
            run_step("Installing PyTorch (latest)",
                     venv_pip + ["install", "torch", "torchaudio",
                                 "--index-url", torch_index])

        run_step("Installing torchao for quantization",
                 venv_pip + ["install", "torchao"],
                 required=False)

        run_step("Installing VibeVoice package",
                 venv_pip + ["install", "-e", str(SCRIPT_DIR)])

        run_step("Installing API dependencies",
                 venv_pip + ["install", "-r",
                             str(SCRIPT_DIR / "requirements-api.txt")])

        if errors:
            print("\n  The following steps failed:")
            for err in errors:
                print(f"    - {err}")
            return False
        return True


# ============================================================
# Main wizard
# ============================================================

TOTAL_STEPS = 7


def main():
    print_header("VibeVoice-FastAPI Interactive Installer")
    print("This wizard will configure and set up VibeVoice-FastAPI for you.")
    print("Press Ctrl+C at any time to cancel.\n")

    detected_os_name = detect_os()
    print(f"Detected OS: {detected_os_name}")

    config = {}

    # ── Step 1: Deployment method ──
    print_step(1, TOTAL_STEPS, "Deployment Method")

    docker_available, compose_cmd = check_docker()
    if docker_available:
        print("Docker detected on your system.")
    else:
        print("Docker was not detected on your system.")

    options = [
        ("Docker (recommended)", "Runs in a container — easiest setup, works everywhere"),
        ("Baremetal", "Install directly on your system — more control, requires Python"),
    ]
    if not docker_available:
        options[0] = ("Docker (not detected)", "Docker does not appear to be installed")

    choice = ask_choice("How would you like to deploy?", options, default=1)
    mode = "docker" if choice == 0 else "baremetal"

    if mode == "docker" and not docker_available:
        print("\n  Docker is not available. Please install Docker first, or choose Baremetal.")
        if not ask_yesno("  Continue with Baremetal instead?", default=True):
            print("  Aborting.")
            sys.exit(1)
        mode = "baremetal"

    config["mode"] = mode

    # ── Step 2: GPU detection ──
    print_step(2, TOTAL_STEPS, "GPU Detection")

    gpus = detect_gpus()
    if gpus:
        print(f"Found {len(gpus)} NVIDIA GPU(s):\n")
        for idx, name, vram in gpus:
            print(f"  GPU {idx}: {name} ({vram} MB VRAM)")

        if len(gpus) == 1:
            config["gpu_id"] = gpus[0][0]
            config["device"] = "cuda"
            print(f"\n  Using GPU {gpus[0][0]} ({gpus[0][1]})")
        else:
            gpu_options = [(f"GPU {idx}: {name} ({vram} MB)", "")
                           for idx, name, vram in gpus]
            selected = ask_choice("\nWhich GPU would you like to use?",
                                  gpu_options, default=1)
            config["gpu_id"] = gpus[selected][0]
            config["device"] = "cuda"
    else:
        print("No NVIDIA GPU detected.")
        if detected_os_name == "macos":
            print("  Apple Silicon detected — using MPS (Metal) acceleration.")
            config["device"] = "mps"
            config["gpu_id"] = None
        else:
            print("  WARNING: Running on CPU will be very slow.")
            if ask_yesno("  Continue with CPU mode?", default=True):
                config["device"] = "cpu"
                config["gpu_id"] = None
            else:
                print("  Aborting. Please ensure your NVIDIA drivers are installed.")
                sys.exit(1)

    # ── Step 3: Model selection ──
    print_step(3, TOTAL_STEPS, "Model Selection")

    model_options = [
        ("VibeVoice-1.5B", "~8GB VRAM, fast, single-speaker only"),
        ("VibeVoice-Large-AWQ (recommended)", "~8.4GB VRAM, RTF ~0.7, multi-speaker"),
        ("VibeVoice-Large (FP16)", "~18GB VRAM, RTF ~0.5, multi-speaker"),
        ("Custom model path", "Enter a HuggingFace model ID or local path"),
    ]
    model_choice = ask_choice("Which model would you like to use?",
                              model_options, default=1)
    if model_choice == 0:
        config["model_path"] = "microsoft/VibeVoice-1.5B"
    elif model_choice == 1:
        config["model_path"] = "ncoder-ai/VibeVoice-Large-AWQ"
    elif model_choice == 2:
        config["model_path"] = "rsxdalv/VibeVoice-Large"
    else:
        config["model_path"] = ask_string("Enter model path (HuggingFace ID or local path)")

    # ── Step 4: Voices directory ──
    print_step(4, TOTAL_STEPS, "Voices Directory")

    voices_options = [
        ("Built-in demo voices (default)", "Uses the included demo voice presets"),
        ("Custom directory", "Point to your own directory of voice audio files"),
    ]
    voices_choice = ask_choice("Where are your voice presets?",
                               voices_options, default=1)
    if voices_choice == 0:
        config["voices_dir"] = "./demo/voices"
    else:
        config["voices_dir"] = ask_path("Enter path to voices directory")

    # ── Step 5: Performance options ──
    print_step(5, TOTAL_STEPS, "Performance Options")

    config["torch_compile"] = ask_yesno(
        "Enable torch.compile? (20-50% speedup, slower first request)", default=False)

    quant_options = [
        ("None (default)", "Full precision — best quality, most VRAM"),
        ("INT8", "~40% less VRAM, minimal quality impact"),
        ("INT4", "~60% less VRAM, some quality impact"),
    ]
    quant_choice = ask_choice("Quantization level?", quant_options, default=1)
    if quant_choice == 1:
        config["quantization"] = "int8_torchao"
    elif quant_choice == 2:
        config["quantization"] = "int4_torchao"
    else:
        config["quantization"] = None

    # ── Step 6: API port ──
    print_step(6, TOTAL_STEPS, "API Port")

    while True:
        port_str = ask_string("API port", default="8001")
        try:
            port = int(port_str)
            if 1 <= port <= 65535:
                config["port"] = port
                break
            print("  Port must be between 1 and 65535.")
        except ValueError:
            print("  Please enter a valid port number.")

    # ── Step 7: Docker-only: HuggingFace cache ──
    if mode == "docker":
        print_step(7, TOTAL_STEPS, "HuggingFace Cache (Docker)")
        default_hf = detect_hf_cache()
        print("The HuggingFace model cache is mounted into the container so models")
        print("don't need to be re-downloaded each time the container starts.")
        config["hf_cache_dir"] = ask_path(
            "HuggingFace cache directory", default=default_hf)
    else:
        print_step(7, TOTAL_STEPS, "HuggingFace Cache")
        print("Models will be cached in the default HuggingFace location.")
        print(f"  ({detect_hf_cache()})")
        config["hf_cache_dir"] = None

    # ── Summary ──
    print_header("Configuration Summary")
    print(f"  Deployment:     {mode.capitalize()}")
    print(f"  OS:             {detected_os_name}")
    print(f"  Device:         {config['device']}")
    if config.get("gpu_id") is not None:
        print(f"  GPU:            {config['gpu_id']}")
    print(f"  Model:          {config['model_path']}")
    print(f"  Voices:         {config['voices_dir']}")
    print(f"  torch.compile:  {'Yes' if config['torch_compile'] else 'No'}")
    print(f"  Quantization:   {config.get('quantization') or 'None'}")
    print(f"  API Port:       {config['port']}")
    if config.get("hf_cache_dir"):
        print(f"  HF Cache:       {config['hf_cache_dir']}")
    print()

    if not ask_yesno("Proceed with this configuration?", default=True):
        print("\nAborted. No changes were made.")
        sys.exit(0)

    # ── Generate .env ──
    print_header("Generating Configuration")

    # Warn if .env already exists
    env_path = SCRIPT_DIR / ".env"
    if env_path.exists():
        if ask_yesno(".env already exists. Overwrite?", default=True):
            backup = SCRIPT_DIR / ".env.bak"
            shutil.copy2(env_path, backup)
            print(f"  Backed up existing .env to .env.bak")
        else:
            print("  Keeping existing .env. Skipping config generation.")
            return

    env_path = generate_env(config, mode)
    print(f"  Generated {env_path}")

    # ── Docker: update GPU in docker-compose.yml ──
    if mode == "docker" and config.get("gpu_id") is not None:
        update_docker_gpu(config["gpu_id"])

    # ── Run setup ──
    if mode == "docker":
        success = run_docker_setup(compose_cmd)
    else:
        success = run_baremetal_setup(detected_os_name)

    # ── Final output ──
    print_header("Setup Complete!" if success else "Setup Finished with Errors")

    if mode == "docker":
        if success:
            print("Your VibeVoice API is starting in Docker!")
            print()
            print("Useful commands:")
            print(f"  View logs:     {' '.join(compose_cmd)} logs -f")
            print(f"  Stop:          {' '.join(compose_cmd)} down")
            print(f"  Restart:       {' '.join(compose_cmd)} restart")
        else:
            print("Docker setup encountered errors. Check the output above.")
            print("You can try running manually:")
            print(f"  {' '.join(compose_cmd)} build")
            print(f"  {' '.join(compose_cmd)} up -d")
    else:
        if success:
            print("VibeVoice dependencies have been installed!")
            print()
            print("To start the server:")
            if detected_os_name == "windows":
                print("  start.bat")
            else:
                print("  ./start.sh")
        else:
            print("Setup encountered errors. Check the output above.")

    print()
    print(f"API will be available at: http://localhost:{config['port']}")
    print(f"API docs:                 http://localhost:{config['port']}/docs")
    print(f"Health check:             http://localhost:{config['port']}/health")
    print()


if __name__ == "__main__":
    # When piped via "curl ... | bash", stdin is the pipe, not the terminal.
    # Reopen stdin from /dev/tty (Unix) so input() prompts work.
    if not sys.stdin.isatty():
        try:
            sys.stdin = open("/dev/tty", "r")
        except OSError:
            pass  # Windows or no tty available — stdin stays as-is

    try:
        main()
    except KeyboardInterrupt:
        print("\n\nInstallation cancelled.")
        sys.exit(1)
