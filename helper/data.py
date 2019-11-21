from typing import List, Dict


def filter(list: List[str], dict: Dict) -> List[str]:
    return [item for item in list if item in dict]