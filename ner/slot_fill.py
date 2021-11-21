import sys    
import logging
import unittest
import re
import requests
import json

logging.basicConfig(level=logging.DEBUG)

boq = 'BOQ'
eoq = 'EOQ'
act_tag = 'ACT'
obj_tag = 'OBJ'
val_tag = 'VAL'


def split_token_tag(token_tag_list):
    """ Input is like
    [{'Double': 'B-OBJ'}, {'Click': 'B-OBJ'}, {'on': 'B-OBJ'},
    Output is like
    tokens = ['Double', 'Click', 'on', 'a', 'calendar',...
    tags = ['B-OBJ', 'B-OBJ', 'B-OBJ', 'B-OBJ', ...
    """
    tokens = [list(item.keys())[0] for item in  token_tag_list]
    tags = [list(item.values())[0] for item in  token_tag_list]
    return tokens, tags


def ner_slot_filling(tokens, ner_tags, raw_outs=None, correct_with_quotes=True):
    """
    Parameters
    tokens : list, a tokenized text
    ner_tags : list, output of the main NER model
    raw_outputs : list, output of the NER model for commands
    correct_with_quotes : bool, correction of entity names inside quotes.
        For instance, if the NER model for input text ['BOQ', 'User', 'name', 'EOQ']
        detects {'OBJ', 'User'}, we can correct it using tokens 'BOQ' and 'EOQ'.
        As result, we return {'OBJ': 'User name'}.

    Returns
    slots : dict. An example is {'cmd_name': 'VERIFY_XPATH', 'ACT': 'verify'}
    """
    #tags = ner_tags

    logging.info('tokens: {}'.format(tokens))
    logging.info('tags: {}'.format(ner_tags))

    tags_set = filter(lambda tag: tag != 'O', ner_tags)
    tags_set = filter(lambda tag: tag[0] == 'B', tags_set)
    tags_set = {tag[2:] for tag in tags_set}
    print('tags_set:', tags_set)

    slots = {}
    # Extract the rest entities
    for tag in tags_set:
        B_tag = 'B-' + tag
        I_tag = 'I-' + tag

        if B_tag in ner_tags:
            B_tag_index = ner_tags.index(B_tag)
            index = B_tag_index + 1
            while index < len(tokens) and ner_tags[index] == I_tag:
                index += 1
            last_I_tag_index = index # the index of the last "I-" tag
            
            if correct_with_quotes:
                left_tokens = tokens[:last_I_tag_index]
                boqs = left_tokens.count(boq)
                eoqs = left_tokens.count(eoq)
                if boqs > eoqs:
                    try:
                        last_I_tag_index = next(i for i, token in enumerate(tokens) if token == eoq and i >= B_tag_index)
                    except:
                        last_I_tag_index = len(tokens) - 1  # if EOQ is absent            
                
            entity_name = " ".join(tokens[B_tag_index: last_I_tag_index])
            logging.debug("entity_name: {}".format(entity_name))
        else:
            entity_name = None

        if type(entity_name) is str:
            entity_name = entity_name.strip()   # .lstrip(boq).rstrip(eoq).strip()
        else:
            logging.warning('The type of entity_name is not str')

        print("entity_name:", entity_name)
        slots[tag] = entity_name

    return slots


#def correct_ner_slot_filling(slots, tokens, ner_tags):
#    return slots



class Test_ner_slot_filling(unittest.TestCase):

    def test_1(self):
        tokens = ['open', 'the', 'website', 'google.com']
        ner_tags = ['B-ACT', 'O', 'B-OBJ', 'B-VAL']
        cmd_tags = ['B-Open', 'O', 'O', 'O']

        slots = ner_slot_filling(tokens, ner_tags)
        print("slots['ACT']: {}".format(slots.get(act_tag)))
        print("slots['OBJ']: {}".format(slots.get(obj_tag)))
        print("slots['VAL']: {}".format(slots.get(val_tag)))

        self.assertEqual(slots.get(act_tag), tokens[0])
        self.assertEqual(slots.get(obj_tag), tokens[2])
        self.assertEqual(slots.get(val_tag), tokens[3])


    def test_1(self):
        tokens = ['click', 'on', boq, 'try', 'for', 'free', eoq, '.']
        ner_tags = ['B-ACT', 'O', 'B-OBJ', 'I-OBJ', 'I-OBJ', 'I-OBJ', 'I-OBJ', 'O']
        cmd_tags = ['B-Click', 'O', 'O', 'O', 'O', 'O', 'O', 'O']

        slots = ner_slot_filling(tokens, ner_tags)
        print("slots['ACT']: {}".format(slots.get(act_tag)))
        print("slots['OBJ']: {}".format(slots.get(obj_tag)))
        print("slots['VAL']: {}".format(slots.get(val_tag)))

        self.assertEqual(slots.get(act_tag), "click")
        self.assertEqual(slots.get(obj_tag), "try for free")
        self.assertEqual(slots.get(val_tag), None)


    def test_LOC(self):
        tokens = ['click', 'on', boq, 'try', 'for', 'free', eoq, 'on', 'page',  '.']
        ner_tags = ['B-ACT', 'O', 'B-OBJ', 'I-OBJ', 'I-OBJ', 'I-OBJ', 'I-OBJ', 'O', 'B-LOC', 'O']
        cmd_tags = ['B-Click', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O']

        slots = ner_slot_filling(tokens, ner_tags)
        print("slots['ACT']: {}".format(slots.get(act_tag)))
        print("slots['OBJ']: {}".format(slots.get(obj_tag)))
        print("slots['VAL']: {}".format(slots.get(val_tag)))
        print("slots['LOC']: {}".format(slots.get('LOC')))

        self.assertEqual(slots.get('ACT'), "click")
        self.assertEqual(slots.get('OBJ'), "try for free")
        self.assertEqual(slots.get('VAL'), None)
        self.assertEqual(slots.get('LOC'), "page")


    def test_Blocks_If(self):
        tokens = ['.', 'if', 'login', 'is', 'on', 'the', 'screen', ',', 'click', 'on', 'login', '.']
        ner_tags = ['O', 'B-ACT', 'B-COND', 'I-COND', 'I-COND', 'I-COND', 'I-COND', 'O', 'B-SUB', 'I-SUB', 'I-SUB', 'O']
        cmd_tags = ['O', 'B-IFELSE', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O']

        slots = ner_slot_filling(tokens, ner_tags)
        print("slots['ACT']: {}".format(slots.get('ACT')))
        print("slots['COND']: {}".format(slots.get('COND')))
        print("slots['SUB']: {}".format(slots.get('SUB')))
        print("slots['cmd_name']: {}".format(slots.get('cmd_name')))

        self.assertEqual(slots.get('ACT'), "if")
        self.assertEqual(slots.get('cmd_name'), "ifelse")
        self.assertEqual(slots.get('COND'), "login is on the screen")
        self.assertEqual(slots.get('SUB'), "click on login")
        #self.assertEqual(slots.get('action_name'), "if")

    def test_Scripts(self):
        tokens = ['.', 'Exec', 'BASHSCRIPT', 'with', 'var1', 'returning', 'var2', '.']
        ner_tags = ['O', 'B-ACT', 'B-OBJ', 'O', 'B-VAR', 'O', 'B-VAL', 'O']
        cmd_tags = ['O', 'B-IF_ELSE', 'B-DOWNLOAD_PDF', 'B-DOWNLOAD_PDF', 'B-DOWNLOAD_PDF', 'O', 'O', 'I-VERIFY']

        slots = ner_slot_filling(tokens, ner_tags)['slots']
        print("slots: {}".format(slots))

        #self.assertEqual(slots.get('ACT'), "exec")
        #self.assertEqual(slots.get('cmd_name'), "exec")


    def test_HOVER(self):
        tokens = ['.', 'hover', 'on', 'BOQ', 'User', 'name', 'EOQ', '.']
        ner_tags = ['O', 'B-ACT', 'I-ACT', 'O', 'B-OBJ', 'O', 'O', 'O']
        cmd_tags = ['O', 'B-BEGIN', 'I-VERIFY', 'O', 'O', 'O', 'O', 'O']
        slots = ner_slot_filling(tokens, ner_tags, correct_with_quotes=True)
        print("slots: {}".format(slots))
        self.assertEqual(slots.get('ACT'), "hover on")
        self.assertEqual(slots.get('OBJ'), "User name")


    def test_HOVER_missing_EOQ(self):
        tokens = ['.', 'hover', 'on', 'BOQ', 'User', 'name', '.']
        ner_tags = ['O', 'B-ACT', 'I-ACT', 'O', 'B-OBJ', 'O', 'O', ]
        cmd_tags = ['O', 'B-BEGIN', 'I-VERIFY', 'O', 'O', 'O', 'O', ]
        slots = ner_slot_filling(tokens, ner_tags, correct_with_quotes=True)
        print("slots: {}".format(slots))
        self.assertEqual(slots.get('ACT'), "hover on")
        self.assertEqual(slots.get('OBJ'), "User name")

if __name__ == '__main__':

    #unittest.main()

    test = Test_ner_slot_filling()
    test.test_HOVER()
    test.test_HOVER_missing_EOQ()

    token_tag_list = [{'Double': 'B-OBJ'}, {'Click': 'B-OBJ'}, 
        {'on': 'B-OBJ'}, {'a': 'B-OBJ'}, {'calendar': 'B-ACT'}, 
        {'from': 'B-OBJ'}, {'the': 'B-OBJ'}, {'list': 'B-OBJ'}, 
        {'on': 'B-OBJ'}, {'the': 'B-OBJ'}, {'left': 'B-OBJ'}, 
        {'side': 'B-OBJ'}, {'of': 'B-OBJ'}, {'the': 'B-OBJ'}, 
        {'screen.': 'B-OBJ'}]

    tokens, tags = split_token_tag(token_tag_list)
    print("tokens:", tokens)
    print("tags:", tags)