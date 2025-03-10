import pandas as pd
import re

# load the dataset
data = pd.read_csv("./demo_dataset.csv")
print(data.head())
print(data.info())
print(data.isnull().sum())

def is_valid_ip(ip):
    pattern = re.compile(r'^((25[0-5]|2[0-4][0-9]|[01]?[0-9][0-9]?)\.){3}(25[0-5]|2[0-4][0-9]|[01]?[0-9][0-9]?)$')
    return bool(pattern.match(ip))

# Check for valid IP addresses
invalid_ips = data[~data['source_ip'].astype(str).apply(is_valid_ip)]
# print(invalid_ips)

def is_valid_port(port):
    try:
        port = int(port)
        return 0 <= port <= 65535
    except ValueError:
        return False
    
# Check for invalid port numbers
invalid_ports = data[~data['destination_port'].apply(is_valid_port)]
# print(invalid_ports)

valid_protocols = ['TCP', 'TLS', 'SSH', 'POP3', 'DNS', 'HTTPS', 'SMTP', 'FTP', 'UDP', 'HTTP']

# check for invalid protocol values
invalid_protocols = data[~data['protocol'].isin(valid_protocols)]
# print(invalid_protocols)

def is_valid_bytes(bytes):
    try:
        bytes = int(bytes)
        return bytes >= 0
    except ValueError:
        return False

# Check for invalid bytes transferred
invalid_bytes = data[~data['bytes_transferred'].apply(is_valid_bytes)]
# print(invalid_bytes)

def is_valid_threat_level(threat_level):
    try:
        threat_level = int(threat_level)
        return 0 <= threat_level <= 2
    except ValueError:
        return False

# Check for invalid threat levels
invalid_threat_levels = data[~data['threat_level'].apply(is_valid_threat_level)]
# print(invalid_threat_levels)

# the ignore errors covers the fact that there might be some overlap between indexes that match other invalid criterias
data = data.drop(invalid_ips.index, errors='ignore')
data = data.drop(invalid_ports.index, errors='ignore')
data = data.drop(invalid_protocols.index, errors='ignore')
data = data.drop(invalid_bytes.index, errors='ignore')
data = data.drop(invalid_threat_levels.index, errors='ignore')

print(data.describe(include='all'))