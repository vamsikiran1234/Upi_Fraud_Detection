#!/usr/bin/env python3
"""
Simple test to check if the server is working
"""

import requests
import time

def test_server():
    """Test if server is running"""
    try:
        print("Testing server connection...")
        response = requests.get("http://localhost:8000", timeout=5)
        print(f"Server response: {response.status_code}")
        print(f"Response: {response.text}")
        return True
    except Exception as e:
        print(f"Connection failed: {e}")
        return False

if __name__ == "__main__":
    test_server()
