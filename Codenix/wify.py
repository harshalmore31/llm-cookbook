# This code is for EDUCATIONAL PURPOSES ONLY and should not be used for malicious activities.
# Use this code responsibly and ethically.
# Sniffing other networks without permission is illegal and unethical.

import socket
import struct
import platform
import socket
import dns.resolver
from collections import defaultdict
import subprocess
import re
import time

def create_raw_socket():
    """Creates a raw socket for capturing network packets."""
    try:
        if platform.system() == "Windows":
            s = socket.socket(socket.AF_INET, socket.SOCK_RAW, socket.IPPROTO_IP)
            s.bind((socket.gethostbyname(socket.gethostname()), 0))
            s.setsockopt(socket.IPPROTO_IP, socket.IP_HDRINCL, 1)
            s.ioctl(socket.SIO_RCVALL, socket.RCVALL_ON)
        elif platform.system() == "Linux":
            s = socket.socket(socket.AF_PACKET, socket.SOCK_RAW, socket.ntohs(0x0003))
        else:
            print("Operating system not supported.")
            return None
        return s
    except socket.error as msg:
        print(f"Socket creation error: {msg}")
        return None

def get_hostname(ip_address):
    """Attempts to get the hostname from an IP address."""
    try:
        answers = dns.resolver.resolve_address(ip_address)
        if answers:
            return str(answers[0])
        return "N/A"
    except (dns.resolver.NXDOMAIN, dns.resolver.NoAnswer, dns.exception.Timeout):
        return "N/A"

def scan_wifi_networks():
    """Scans for nearby Wi-Fi networks."""
    networks = {}
    try:
        if platform.system() == "Windows":
            output = subprocess.check_output(["netsh", "wlan", "show", "networks"], text=True)
            for line in output.splitlines():
                match = re.search(r"SSID\s*:\s*(.+)", line)
                if match:
                   ssid = match.group(1).strip()
                   networks[ssid] = "N/A" 
        elif platform.system() == "Linux":
            output = subprocess.check_output(["iwlist", "wlan0", "scan"], text=True, stderr=subprocess.DEVNULL)
            
            for line in output.splitlines():
                match = re.search(r"ESSID:\"(.+)\"", line)
                if match:
                   ssid = match.group(1).strip()
                   networks[ssid] = "N/A"
        else:
            print("OS not supported.")
            return None
        return networks
    except Exception as e:
        print(f"Error scanning Wi-Fi networks: {e}")
        return None

def analyze_packet(packet, os_type, source_info, networks_data):
    """Analyzes a captured packet and prints key information."""
    if os_type == "Windows":
        ip_header = packet[0:20]
        ip_version = (ip_header[0] >> 4)
        ip_header_length = (ip_header[0] & 0x0F) * 4
        ip_protocol = ip_header[9]
        ip_src_addr = socket.inet_ntoa(ip_header[12:16])
        ip_dest_addr = socket.inet_ntoa(ip_header[16:20])

        if ip_src_addr not in source_info:
             source_info[ip_src_addr] = get_hostname(ip_src_addr)
        
        print(f"Source Host Name: {source_info.get(ip_src_addr,'N/A')}")

        print(f"IP Header:")
        print(f"  Version: {ip_version}")
        print(f"  Header Length: {ip_header_length}")
        print(f"  Protocol: {ip_protocol}")
        print(f"  Source IP: {ip_src_addr}")
        print(f"  Destination IP: {ip_dest_addr}")

        if ip_protocol == 6: # TCP
            tcp_header = packet[ip_header_length:ip_header_length + 20]
            tcp_src_port = struct.unpack("!H", tcp_header[0:2])[0]
            tcp_dest_port = struct.unpack("!H", tcp_header[2:4])[0]
            print(f"    TCP Header:")
            print(f"      Source Port: {tcp_src_port}")
            print(f"      Destination Port: {tcp_dest_port}")

        elif ip_protocol == 17: # UDP
            udp_header = packet[ip_header_length:ip_header_length + 8]
            udp_src_port = struct.unpack("!H", udp_header[0:2])[0]
            udp_dest_port = struct.unpack("!H", udp_header[2:4])[0]
            print(f"    UDP Header:")
            print(f"      Source Port: {udp_src_port}")
            print(f"      Destination Port: {udp_dest_port}")

        if len(packet) > ip_header_length + 20:  # Check if there's payload (simplistic)
             payload = packet[ip_header_length + 20:]
             if payload:
                  try:
                      decoded_payload = payload.decode('utf-8', errors='ignore').strip()
                      if decoded_payload:
                           print(f"   Payload (attempted decode): {decoded_payload[:200]}...")
                  except Exception:
                       print(f"   Payload: Non-text or unreadable")
    elif os_type == "Linux":
        ethernet_header = packet[0:14]
        ethernet_dest_mac = ethernet_header[0:6].hex()
        ethernet_src_mac = ethernet_header[6:12].hex()
        ethertype = struct.unpack("!H", ethernet_header[12:14])[0]

        if ethertype == 0x0800: # IPv4
            ip_header = packet[14:34]
            ip_version = (ip_header[0] >> 4)
            ip_header_length = (ip_header[0] & 0x0F) * 4
            ip_protocol = ip_header[9]
            ip_src_addr = socket.inet_ntoa(ip_header[12:16])
            ip_dest_addr = socket.inet_ntoa(ip_header[16:20])

            if ip_src_addr not in source_info:
                 source_info[ip_src_addr] = get_hostname(ip_src_addr)
            
            print(f"Source Host Name: {source_info.get(ip_src_addr,'N/A')}")
            print(f"  Ethernet Header:")
            print(f"  Destination MAC: {ethernet_dest_mac}")
            print(f"  Source MAC: {ethernet_src_mac}")
            print(f"  EtherType: {ethertype}")
            print(f"  IP Header:")
            print(f"    Version: {ip_version}")
            print(f"    Header Length: {ip_header_length}")
            print(f"    Protocol: {ip_protocol}")
            print(f"    Source IP: {ip_src_addr}")
            print(f"    Destination IP: {ip_dest_addr}")
            
            if ip_protocol == 6: # TCP
                 tcp_header = packet[14 + ip_header_length:14 + ip_header_length + 20]
                 tcp_src_port = struct.unpack("!H", tcp_header[0:2])[0]
                 tcp_dest_port = struct.unpack("!H", tcp_header[2:4])[0]
                 print(f"    TCP Header:")
                 print(f"       Source Port: {tcp_src_port}")
                 print(f"       Destination Port: {tcp_dest_port}")

            elif ip_protocol == 17: # UDP
                udp_header = packet[14 + ip_header_length:14+ip_header_length +8]
                udp_src_port = struct.unpack("!H", udp_header[0:2])[0]
                udp_dest_port = struct.unpack("!H", udp_header[2:4])[0]
                print(f"   UDP Header:")
                print(f"       Source Port: {udp_src_port}")
                print(f"       Destination Port: {udp_dest_port}")
            if len(packet) > 14 + ip_header_length + 20:  # Check if there's payload (simplistic)
                 payload = packet[14+ ip_header_length + 20:]
                 if payload:
                      try:
                         decoded_payload = payload.decode('utf-8', errors='ignore').strip()
                         if decoded_payload:
                            print(f"  Payload (attempted decode): {decoded_payload[:200]}...")
                      except Exception:
                         print(f"   Payload: Non-text or unreadable")

def capture_packets(socket):
    """Captures packets from the network and analyzes them."""
    try:
        source_info = defaultdict(str)
        networks_data = defaultdict(str)
        if platform.system() == "Windows":
            os_type = "Windows"
        elif platform.system() == "Linux":
            os_type = "Linux"
        else:
            print("OS not supported")
            return
        
        scanned_networks = scan_wifi_networks()
        if scanned_networks:
             print("Available Wi-Fi Networks:")
             for ssid in scanned_networks:
                  print(f"   - {ssid}")
        else:
            print("No wi-fi networks found")

        while True:
            packet = socket.recvfrom(65535)[0]
            analyze_packet(packet, os_type, source_info, networks_data)
            time.sleep(0.1)  # Add a small delay to avoid overwhelming output
    except KeyboardInterrupt:
        print("Packet capture stopped.")
    except socket.error as msg:
        print(f"Socket receive error: {msg}")

if __name__ == "__main__":
    raw_socket = create_raw_socket()
    if raw_socket:
        capture_packets(raw_socket)