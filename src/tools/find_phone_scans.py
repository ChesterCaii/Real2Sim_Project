#!/usr/bin/env python3
"""
Find Phone Scan Files
Helps locate OBJ files from phone scanning apps
"""

import os
import glob
from pathlib import Path

def find_obj_files():
    """Find all OBJ files on the system that might be phone scans"""
    print("Searching for OBJ files...")
    print("=" * 40)
    
    # Common locations where phone apps save files
    search_paths = [
        # Current directory and Downloads
        ".",
        "~/Downloads",
        "~/Desktop", 
        "~/Documents",
        
        # iOS Files app locations
        "~/Library/Mobile Documents",
        "~/Library/Application Support",
        
        # Common app directories
        "~/Documents/3D Scanner App",
        "~/Documents/Scaniverse",
        "~/Documents/Polycam",
        
        # Recent files
        "~/Downloads/*.obj",
        "~/Desktop/*.obj",
        "~/Documents/*.obj"
    ]
    
    found_files = []
    
    for path_pattern in search_paths:
        try:
            # Expand user path
            expanded_path = os.path.expanduser(path_pattern)
            
            if "*" in expanded_path:
                # Use glob for wildcard patterns
                files = glob.glob(expanded_path)
                for file in files:
                    if file.endswith('.obj'):
                        found_files.append(file)
            else:
                # Search directory for OBJ files
                if os.path.isdir(expanded_path):
                    for root, dirs, files in os.walk(expanded_path):
                        for file in files:
                            if file.lower().endswith('.obj'):
                                full_path = os.path.join(root, file)
                                found_files.append(full_path)
        except Exception as e:
            continue  # Skip inaccessible paths
    
    # Remove duplicates and sort
    found_files = list(set(found_files))
    found_files.sort(key=lambda x: os.path.getmtime(x), reverse=True)  # Most recent first
    
    if found_files:
        print(f"Found {len(found_files)} OBJ files:")
        print()
        for i, file_path in enumerate(found_files[:10]):  # Show top 10
            file_size = os.path.getsize(file_path)
            mod_time = os.path.getmtime(file_path)
            import time
            mod_time_str = time.strftime('%Y-%m-%d %H:%M', time.localtime(mod_time))
            
            print(f"{i+1}. {os.path.basename(file_path)}")
            print(f"   Path: {file_path}")
            print(f"   Size: {file_size} bytes")
            print(f"   Modified: {mod_time_str}")
            print()
        
        return found_files[:10]
    else:
        print("No OBJ files found.")
        print()
        print("This means you probably need to:")
        print("1. Scan an object with a phone app first")
        print("2. Export/save it as OBJ format")
        print("3. Transfer it to your computer if scanned on phone")
        return []

def transfer_instructions():
    """Show how to transfer files from phone to computer"""
    print("How to Transfer Scan from Phone to Computer")
    print("=" * 50)
    print()
    
    print("METHOD 1: AirDrop (iPhone to Mac):")
    print("1. In scanning app, find 'Share' or 'Export' option")
    print("2. Choose 'AirDrop'") 
    print("3. Select your Mac")
    print("4. File will appear in Downloads folder")
    print()
    
    print("METHOD 2: Email/Messages:")
    print("1. Export scan from app")
    print("2. Share via Email or Messages to yourself")
    print("3. Download attachment on computer")
    print()
    
    print("METHOD 3: Cloud Storage:")
    print("1. Export to iCloud Drive/Google Drive/Dropbox")
    print("2. Access from computer")
    print()
    
    print("METHOD 4: Direct USB (Android):")
    print("1. Connect phone to computer via USB")
    print("2. Find files in phone storage")
    print("3. Copy to computer")

def quick_test_scan():
    """Create a quick test scan for demo purposes"""
    print("Creating Test Object for Demo")
    print("=" * 40)
    
    # Create a simple test object (teapot shape)
    test_obj = """# Test Teapot for Real2Sim Demo
# Generated for phone scan simulation
v -0.05 -0.05 0.0
v 0.05 -0.05 0.0
v 0.05 0.05 0.0
v -0.05 0.05 0.0
v -0.03 -0.03 0.08
v 0.03 -0.03 0.08
v 0.03 0.03 0.08
v -0.03 0.03 0.08
v 0.0 0.0 0.12
v 0.08 0.0 0.04
v 0.08 0.02 0.06

# Base
f 1 2 3 4
# Sides
f 1 5 6 2
f 2 6 7 3
f 3 7 8 4
f 4 8 5 1
# Top (partial)
f 5 6 9
f 6 7 9
f 7 8 9
f 8 5 9
# Spout
f 6 7 10
f 7 10 11
f 10 11 6
"""
    
    test_path = "test_phone_scan.obj"
    with open(test_path, 'w') as f:
        f.write(test_obj)
    
    print(f"✓ Created test scan: {test_path}")
    print("✓ Use this to test the system while you prepare a real scan")
    
    return os.path.abspath(test_path)

def main():
    print("Real2Sim Phone Scan File Finder")
    print("=" * 40)
    
    while True:
        print("\nOptions:")
        print("1. Search for existing OBJ files on this computer")
        print("2. Show how to transfer files from phone") 
        print("3. Create test object for demo (while you prepare real scan)")
        print("4. Exit")
        
        choice = input("\nChoice: ").strip()
        
        if choice == "1":
            found_files = find_obj_files()
            if found_files:
                print("Copy one of these paths to use in the phone scanner guide!")
        elif choice == "2":
            transfer_instructions()
        elif choice == "3":
            test_path = quick_test_scan()
            print(f"\nTest file created! Use this path in scanner guide:")
            print(f"{test_path}")
        elif choice == "4":
            break
        else:
            print("Invalid choice")

if __name__ == "__main__":
    main() 