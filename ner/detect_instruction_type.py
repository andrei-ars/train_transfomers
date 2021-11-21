#import os
from ner_slot_filling import split_token_tag


def remove_prefix(tag):
    return tag[2:] if tag != 'O' else tag

def detect_instruction_type(tokens, ner_tags):
    TYP = "TYP"
    core_tags = [remove_prefix(tag) for tag in ner_tags]
    print(core_tags)
    if TYP in core_tags:
        index = core_tags.index(TYP)
        if tokens[index].upper() == "WHERE":
            return "WHERE"

    return 0

if __name__ == "__main__":
    token_tag_list = [{'enter': 'B-ACT'}, {'username': 'B-OBJ'}, {'where': 'B-TYP'}, {'age': 'B-CNT'}, {'=': 'B-OPE'}, {'20': 'B-CNT'}]
    tokens, tags = split_token_tag(token_tag_list)
    print(tokens)
    print(tags)
    r = detect_instruction_type(tokens, tags)
    print(r)








