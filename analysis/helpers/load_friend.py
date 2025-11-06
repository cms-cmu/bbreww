from typing import TypedDict 
import awkward as ak 
from src.data_formats.root import Chunk, Friend

class FriendTemplate(TypedDict):
    path: str
    keys: str | list[dict[str]]


def parse_friends(args: dict[str, str | FriendTemplate]) -> dict[str, Friend]:
    friends = {}
    if args is None:
        return friends

    from src.classifier.task import parse

    for name, path in args.items():
        if isinstance(path, str):
            friends[name] = Friend.from_json(parse.mapping(path, "file"))
        else:
            keys = path["keys"]
            if isinstance(keys, str):
                keys = eval(keys)
            for key in keys:
                friends[name.format(**key)] = Friend.from_json(
                    parse.mapping(path["path"].format(**key), "file")
                )

    return friends
