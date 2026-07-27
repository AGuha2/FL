from typing import Final

import pandas as pd
from sklearn.preprocessing import LabelEncoder


ORIGINAL_LABELS: Final[list[str]] = [
    "BACKDOOR_MALWARE",
    "BENIGN",
    "BROWSERHIJACKING",
    "COMMANDINJECTION",
    "DDOS-ACK_FRAGMENTATION",
    "DDOS-HTTP_FLOOD",
    "DDOS-ICMP_FLOOD",
    "DDOS-ICMP_FRAGMENTATION",
    "DDOS-PSHACK_FLOOD",
    "DDOS-RSTFINFLOOD",
    "DDOS-SLOWLORIS",
    "DDOS-SYN_FLOOD",
    "DDOS-SYNONYMOUSIP_FLOOD",
    "DDOS-TCP_FLOOD",
    "DDOS-UDP_FLOOD",
    "DDOS-UDP_FRAGMENTATION",
    "DICTIONARYBRUTEFORCE",
    "DNS_SPOOFING",
    "DOS-HTTP_FLOOD",
    "DOS-SYN_FLOOD",
    "DOS-TCP_FLOOD",
    "DOS-UDP_FLOOD",
    "MITM-ARPSPOOFING",
    "MIRAI-GREETH_FLOOD",
    "MIRAI-GREIP_FLOOD",
    "MIRAI-UDPPLAIN",
    "RECON-HOSTDISCOVERY",
    "RECON-OSSCAN",
    "RECON-PINGSWEEP",
    "RECON-PORTSCAN",
    "SQLINJECTION",
    "UPLOADING_ATTACK",
    "VULNERABILITYSCAN",
    "XSS",
]


ALL_LABELS: Final[list[str]] = [
    "BENIGN",
    "BRUTEFORCE",
    "DDOS",
    "DOS",
    "MIRAI",
    "RECON",
    "SPOOFING",
    "WEB",
]


LABEL_TO_FAMILY: Final[dict[str, str]] = {
    "BENIGN": "BENIGN",

    "DDOS-ACK_FRAGMENTATION": "DDOS",
    "DDOS-HTTP_FLOOD": "DDOS",
    "DDOS-ICMP_FLOOD": "DDOS",
    "DDOS-ICMP_FRAGMENTATION": "DDOS",
    "DDOS-PSHACK_FLOOD": "DDOS",
    "DDOS-RSTFINFLOOD": "DDOS",
    "DDOS-SLOWLORIS": "DDOS",
    "DDOS-SYN_FLOOD": "DDOS",
    "DDOS-SYNONYMOUSIP_FLOOD": "DDOS",
    "DDOS-TCP_FLOOD": "DDOS",
    "DDOS-UDP_FLOOD": "DDOS",
    "DDOS-UDP_FRAGMENTATION": "DDOS",

    "DOS-HTTP_FLOOD": "DOS",
    "DOS-SYN_FLOOD": "DOS",
    "DOS-TCP_FLOOD": "DOS",
    "DOS-UDP_FLOOD": "DOS",

    "MIRAI-GREETH_FLOOD": "MIRAI",
    "MIRAI-GREIP_FLOOD": "MIRAI",
    "MIRAI-UDPPLAIN": "MIRAI",

    "RECON-HOSTDISCOVERY": "RECON",
    "RECON-OSSCAN": "RECON",
    "RECON-PINGSWEEP": "RECON",
    "RECON-PORTSCAN": "RECON",
    "VULNERABILITYSCAN": "RECON",

    "DNS_SPOOFING": "SPOOFING",
    "MITM-ARPSPOOFING": "SPOOFING",

    "BACKDOOR_MALWARE": "WEB",
    "BROWSERHIJACKING": "WEB",
    "COMMANDINJECTION": "WEB",
    "SQLINJECTION": "WEB",
    "UPLOADING_ATTACK": "WEB",
    "XSS": "WEB",

    "DICTIONARYBRUTEFORCE": "BRUTEFORCE",
}


def convert_to_family(label: object) -> str:
    normalized = str(label).strip().upper()

    if normalized not in LABEL_TO_FAMILY:
        raise ValueError(
            f"Unknown CICIoT2023 label: {normalized!r}"
        )

    return LABEL_TO_FAMILY[normalized]


def convert_series_to_family(series: pd.Series) -> pd.Series:
    normalized = series.astype(str).str.strip().str.upper()

    unknown = sorted(
        set(normalized.unique()) - set(LABEL_TO_FAMILY)
    )

    if unknown:
        raise ValueError(
            f"Unknown labels found: {unknown}"
        )

    return normalized.map(LABEL_TO_FAMILY)


le = LabelEncoder()
le.fit(ALL_LABELS)

NUM_CLASSES: Final[int] = len(ALL_LABELS)
NUM_FEATURES: Final[int] = 39


missing_mappings = sorted(
    set(ORIGINAL_LABELS) - set(LABEL_TO_FAMILY)
)

if missing_mappings:
    raise RuntimeError(
        f"Original labels without mappings: {missing_mappings}"
    )