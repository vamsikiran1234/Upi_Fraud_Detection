# Docker Installation Guide for Windows

## Install Docker Desktop for Windows

1. **Download Docker Desktop**
   - Go to: https://www.docker.com/products/docker-desktop/
   - Download Docker Desktop for Windows
   - Run the installer as Administrator

2. **System Requirements**
   - Windows 10 64-bit: Pro, Enterprise, or Education (Build 15063 or later)
   - Windows 11 64-bit: Home or Pro
   - WSL 2 feature enabled
   - Virtualization enabled in BIOS

3. **Enable WSL 2**
   ```powershell
   # Run in PowerShell as Administrator
   dism.exe /online /enable-feature /featurename:Microsoft-Windows-Subsystem-Linux /all /norestart
   dism.exe /online /enable-feature /featurename:VirtualMachinePlatform /all /norestart
   ```

4. **Restart your computer**

5. **Install WSL 2 Linux kernel**
   - Download: https://wslstorestorage.blob.core.windows.net/wslblob/wsl_update_x64.msi
   - Install the package

6. **Set WSL 2 as default**
   ```powershell
   wsl --set-default-version 2
   ```

7. **Install Docker Desktop**
   - Run the downloaded installer
   - Follow the setup wizard
   - Enable "Use WSL 2 based engine"

8. **Verify Installation**
   ```powershell
   docker --version
   docker-compose --version
   ```

## After Installation

Once Docker is installed, you can run:
```bash
python start_system.py
```
