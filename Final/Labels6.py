from typing import Final
import pandas as pd
from sklearn.preprocessing import LabelEncoder

ALL_LABELS: Final[list[str]] = [
    "BENIGN", "DDOS", "DOS", "MIRAI",
    "RECON", "OTHER",
]

LABEL_TO_FAMILY: Final[dict[str, str]] = {
    "BENIGN": "BENIGN",
    "DICTIONARYBRUTEFORCE": "OTHER",
    "DDOS-ACK_FRAGMENTATION": "DDOS", "DDOS-HTTP_FLOOD": "DDOS",
    "DDOS-ICMP_FLOOD": "DDOS", "DDOS-ICMP_FRAGMENTATION": "DDOS",
    "DDOS-PSHACK_FLOOD": "DDOS", "DDOS-RSTFINFLOOD": "DDOS",
    "DDOS-SLOWLORIS": "DDOS", "DDOS-SYN_FLOOD": "DDOS",
    "DDOS-SYNONYMOUSIP_FLOOD": "DDOS", "DDOS-TCP_FLOOD": "DDOS",
    "DDOS-UDP_FLOOD": "DDOS", "DDOS-UDP_FRAGMENTATION": "DDOS",
    "DOS-HTTP_FLOOD": "DOS", "DOS-SYN_FLOOD": "DOS",
    "DOS-TCP_FLOOD": "DOS", "DOS-UDP_FLOOD": "DOS",
    "MIRAI-GREETH_FLOOD": "MIRAI", "MIRAI-GREIP_FLOOD": "MIRAI",
    "MIRAI-UDPPLAIN": "MIRAI",
    "RECON-HOSTDISCOVERY": "RECON", "RECON-OSSCAN": "RECON",
    "RECON-PINGSWEEP": "RECON", "RECON-PORTSCAN": "RECON",
    "VULNERABILITYSCAN": "RECON",
    "DNS_SPOOFING": "OTHER", "MITM-ARPSPOOFING": "OTHER",
    "BACKDOOR_MALWARE": "OTHER", "BROWSERHIJACKING": "OTHER",
    "COMMANDINJECTION": "OTHER", "SQLINJECTION": "OTHER",
    "UPLOADING_ATTACK": "OTHER", "XSS": "OTHER",
}


def convert_series_to_family(series: pd.Series) -> pd.Series:
    normalized = series.astype(str).str.strip().str.upper()
    unknown = sorted(set(normalized.unique()) - set(LABEL_TO_FAMILY))
    if unknown:
        raise ValueError(f"Unknown labels found: {unknown}")
    return normalized.map(LABEL_TO_FAMILY)


le = LabelEncoder().fit(ALL_LABELS)
NUM_CLASSES: Final[int] = len(ALL_LABELS)
NUM_FEATURES: Final[int] = 39
