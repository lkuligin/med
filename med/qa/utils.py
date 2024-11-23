from typing import Dict, Union

from langchain_core.messages import BaseMessage


def _format_options(entry: Dict[str, str]) -> str:
    return "\n".join([f"{value['key']}: {value['value']}" for value in entry])


def _parse_response(response: Union[BaseMessage, str]) -> str:
    if isinstance(response, str):
        result = response
    else:
        result = response.content
    result = result.strip(".:[\"'(\\ *\n `")
    if result:
        return result[0].upper()
    return "N"
