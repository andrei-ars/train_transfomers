uimport sys    
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


def split_dict_list_to_tokens_and_tags(dict_list):
    tokens = [list(d.keys())[0] for d in dict_list]
    ner_tags = [list(d.values())[0] for d in dict_list]
    return tokens, ner_tags


def remove_prefix(tag):
    return tag[2:] if tag != 'O' else tag


def ner_slot_filling_compound(tokens, ner_tags, cmd_tags=None, correct_with_quotes=True):
    """
    For compound instrucitons.

    Example
    Input:
        tokens:   ['Left', 'click', 'on', 'Browse', 'and', 'Windows', 'and', 'Enter', 'text', 'in', 'document', 'single', 'file']
        ner_tags: ['B-ACT', 'I-ACT', 'O', 'B-OBJ', 'O', 'B-OBJ', 'O', 'B-ACT', 'O', 'O', 'B-OBJ', 'I-OBJ', 'I-OBJ']

    Output:
     {'text_results': 
        [{'ACT': 'Left click', 'OBJ': 'Browse'}, 
        {'ACT': 'Left click', 'OBJ': 'Windows'}, 
        {'ACT': 'Enter', 'OBJ': 'document single file'}], 
     'ind_results': 
        [{'ACT': (0, 1), 'OBJ': (3, 3)}, 
        {'ACT': (0, 1), 'OBJ': (5, 5)}, 
        {'ACT': (7, 7), 'OBJ': (10, 12)}]}
    """
    print('tokens: {}'.format(tokens))
    print('ner_tags: {}'.format(ner_tags))

    tags_core = [remove_prefix(tag) for tag in ner_tags]
    print("tags_core:", tags_core)

    nonzero_tags_set = set(filter(lambda tag: tag != 'O', tags_core))
    print(nonzero_tags_set)
    #tags_set = filter(lambda tag: tag[0] == 'B', tags_set)
    #tags_set = {remove_prefix(tag) for tag in tags_set}
    print('nonzero_tags_set:', nonzero_tags_set)

    entity = {}
    b_entity = {}
    e_entity = {}
    for t in nonzero_tags_set:
        entity[t] = []
        b_entity[t] = []
        e_entity[t] = []
        
        for i, tag in enumerate(tags_core):
            if tag == t and ( (i==0) or (i > 0 and tags_core[i-1] != t) ):
                b_entity[t].append(i)
            if tag == t and ( (i==len(tags_core)-1) or (i < len(tags_core)-1 and tags_core[i+1] != t) ):
                e_entity[t].append(i)

        #print("b_entity:", b_entity)
        #print("e_entity:", e_entity)
        assert len(b_entity[t]) == len(e_entity[t])
        for k in range(len(b_entity[t])):
            entity[t].append( (b_entity[t][k], e_entity[t][k]) )

    print("b_entity:", b_entity)
    print("e_entity:", e_entity)
    print("ACTs:", entity.get('ACT'))
    print("OBJs:", entity.get('OBJ'))

    # For each action we find its objects
    num_actions = len(b_entity['ACT']) if b_entity.get('ACT') else 0
    num_objects = len(b_entity['OBJ']) if b_entity.get('OBJ') else 0  # total number of objects
    actions = []

    for i in range(num_actions):
        actions.append(
            {
                'pos':   entity['ACT'][i],
                'b_pos': b_entity['ACT'][i],
                'objects': [],
            }
        )
        #actions[i]['pos'] = b_entity['ACT'][i]
        #print("i:", i)
        #print("actions[i]['pos']:", actions[i]['pos'])

    for i in range(num_actions):
        #actions[i]['index'] = i
        act_pos = actions[i]['b_pos']
        print("i={}, act_pos={}".format(i, act_pos))
        
        if i < num_actions - 1:
            next_act_pos = actions[i+1]['b_pos']
        else:
            next_act_pos = 100000

        for j in range(num_objects):
            obj_pos = b_entity['OBJ'][j]
            if obj_pos > act_pos and obj_pos < next_act_pos:
                actions[i]['objects'].append(j)

    #logging.debug("actions and indecies of its objects:", actions)
    text_results = []
    ind_results = []
    num_act_obj_pairs = 0

    for i in range(num_actions):
        act_range = actions[i]['pos']
        #print(i)
        #print("range:", list(range(act_pos[0], act_pos[1]+1)))
        act_tokens = tokens[act_range[0]: act_range[1]+1]
        act_phrase = " ".join(act_tokens)
        obj_indices = actions[i]['objects']
        print("i: {}, obj_indices: {}".format(i, obj_indices))
        for j in range(len(obj_indices)):
            obj_index = obj_indices[j]
            obj_range = entity['OBJ'][obj_index]
            obj_tokens = tokens[obj_range[0]: obj_range[1]+1]
            obj_phrase = " ".join(obj_tokens)
            text_result = {"ACT": act_phrase, "OBJ": obj_phrase}
            ind_result = {"ACT": act_range, "OBJ": obj_range}
            text_results.append(text_result)
            ind_results.append(ind_result)
            num_act_obj_pairs += 1
            #print("text_result: {}".format(text_result))
            #print("ind_result: {}".format(ind_result))
        #if len(obj_indices) == 0:
        #    text_results.append(None)
        #    ind_results.append(None)

    return {'num_actions': num_actions, 'num_act_obj_pairs': num_act_obj_pairs,
                'text_results': text_results, 'ind_results': ind_results}


def ner_slot_filling(tokens, ner_tags, cmd_tags, correct_with_quotes=True):
    """
    Parameters
    tokens : list, a tokenized text
    ner_tags : list, output of the main NER model
    cmd_tags : list, output of the NER model for commands
    correct_with_quotes : bool, correction of entity names inside quotes.
        For instance, if the NER model for input text ['BOQ', 'User', 'name', 'EOQ']
        detects {'OBJ', 'User'}, we can correct it using tokens 'BOQ' and 'EOQ'.
        As result, we return {'OBJ': 'User name'}.

    Returns
    slots : dict. An example is {'cmd_name': 'VERIFY_XPATH', 'ACT': 'verify'}
    """

    logging.info('tokens: {}'.format(tokens))
    logging.info('ner_tags: {}'.format(ner_tags))
    logging.info('cmd_tags: {}'.format(cmd_tags))

    tags_set = filter(lambda tag: tag != 'O', ner_tags)
    tags_set = filter(lambda tag: tag[0] == 'B', tags_set)
    tags_set = {tag[2:] for tag in tags_set}
    print('tags_set:', tags_set)

    if act_tag in tags_set:
        if 'B-ACT' in ner_tags:
            # Try to get the cmd_name based on NER tags,
            # i.e. corresponding to the ner-tag B-ACT.
            B_ACT_index = ner_tags.index('B-ACT')
            cmd_tag = cmd_tags[B_ACT_index]
            if cmd_tag != 'O' and "-" in cmd_tag:
                    cmd_name = cmd_tag.split('-')[1]
            else:
                cmd_name = None
        else:
            cmd_name = None

        if cmd_name == None: # (or cmd_tag == 'O')
            # Try to find cmd_name another way.
            non_empty_tags = list(filter(lambda tag: tag != 'O', cmd_tags))
            if len(non_empty_tags) > 0:
                first_tag_index = cmd_tags.index(non_empty_tags[0])
                cmd_name = tokens[first_tag_index] #.lower()
    else:
        cmd_name = None

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

    return {'slots': slots, 'cmd_name': cmd_name}


def correct_ner_slot_filling(slots, tokens, ner_tags):

    return slots



class Test_ner_slot_filling(unittest.TestCase):

    def test_1(self):
        tokens = ['open', 'the', 'website', 'google.com']
        ner_tags = ['B-ACT', 'O', 'B-OBJ', 'B-VAL']
        cmd_tags = ['B-Open', 'O', 'O', 'O']

        slots = ner_slot_filling(tokens, ner_tags, cmd_tags)
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

        slots = ner_slot_filling(tokens, ner_tags, cmd_tags)
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

        slots = ner_slot_filling(tokens, ner_tags, cmd_tags)
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

        slots = ner_slot_filling(tokens, ner_tags, cmd_tags)
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

        slots = ner_slot_filling(tokens, ner_tags, cmd_tags)['slots']
        print("slots: {}".format(slots))

        #self.assertEqual(slots.get('ACT'), "exec")
        #self.assertEqual(slots.get('cmd_name'), "exec")


    def test_HOVER(self):
        tokens = ['.', 'hover', 'on', 'BOQ', 'User', 'name', 'EOQ', '.']
        ner_tags = ['O', 'B-ACT', 'I-ACT', 'O', 'B-OBJ', 'O', 'O', 'O']
        cmd_tags = ['O', 'B-BEGIN', 'I-VERIFY', 'O', 'O', 'O', 'O', 'O']
        slots = ner_slot_filling(tokens, ner_tags, cmd_tags, correct_with_quotes=True)['slots']
        print("slots: {}".format(slots))
        self.assertEqual(slots.get('ACT'), "hover on")
        self.assertEqual(slots.get('OBJ'), "User name")


    def test_HOVER_missing_EOQ(self):
        tokens = ['.', 'hover', 'on', 'BOQ', 'User', 'name', '.']
        ner_tags = ['O', 'B-ACT', 'I-ACT', 'O', 'B-OBJ', 'O', 'O', ]
        cmd_tags = ['O', 'B-BEGIN', 'I-VERIFY', 'O', 'O', 'O', 'O', ]
        slots = ner_slot_filling(tokens, ner_tags, cmd_tags, correct_with_quotes=True)['slots']
        print("slots: {}".format(slots))
        self.assertEqual(slots.get('ACT'), "hover on")
        self.assertEqual(slots.get('OBJ'), "User name")


    def test_compound(self):
        #dict_list = [{'Click': 'B-ACT'}, {'on': 'O'}, {'Browse': 'B-OBJ'}, {'and': 'O'}, 
        #    {'Enter': 'B-ACT'}, {'text': 'O'}, {'in': 'O'}, {'document': 'B-OBJ'}]
        dict_list = [{'Left': 'B-ACT'}, {'click': 'I-ACT'}, {'on': 'O'}, {'Browse': 'B-OBJ'},\
            {'and': 'O'}, {'Windows': 'B-OBJ'}, {'and': 'O'},\
            {'Enter': 'B-ACT'}, {'text': 'O'}, {'in': 'O'}, {'document': 'B-OBJ'}, {'single': 'I-OBJ'}, {'file': 'I-OBJ'}]
        tokens, ner_tags = split_dict_list_to_tokens_and_tags(dict_list)
        #tokens = ['.', 'hover', 'on', 'BOQ', 'User', 'name', '.']
        #ner_tags = ['O', 'B-ACT', 'I-ACT', 'O', 'B-OBJ', 'O', 'O', ]
        results = ner_slot_filling_compound(tokens, ner_tags, correct_with_quotes=False)
        print("results: {}".format(results))


    def test_compound_2(self):
        #dict_list = [{'Click': 'B-ACT'}, {'on': 'O'}, {'Browse': 'B-OBJ'}, {'and': 'O'}, 
        #    {'Enter': 'B-ACT'}, {'text': 'O'}, {'in': 'O'}, {'document': 'B-OBJ'}]
        dict_list = [{'Left': 'B-ACT'}, {'click': 'I-ACT'}, {'on': 'O'}, {'Browse': 'B-OBJ'},\
            {'and': 'O'}, {'Windows': 'B-OBJ'}, {'and': 'O'},\
            {'Enter': 'B-ACT'}, {'text': 'O'}, {'in': 'O'}, {'document': 'B-OBJ'}, {'single': 'I-OBJ'},
            {'file': 'I-OBJ'}, {'Enter': 'B-ACT'}]
        tokens, ner_tags = split_dict_list_to_tokens_and_tags(dict_list)
        #tokens = ['.', 'hover', 'on', 'BOQ', 'User', 'name', '.']
        #ner_tags = ['O', 'B-ACT', 'I-ACT', 'O', 'B-OBJ', 'O', 'O', ]
        results = ner_slot_filling_compound(tokens, ner_tags, correct_with_quotes=False)
        print("results: {}".format(results))

    def test_compound_3(self):
        tokens = ['Click', 'on', 'BOQ', 'INSIDEQOUTES1', 'EOQ']
        ner_tags = ['B-ACT', 'O', 'O', 'O', 'O']
        results = ner_slot_filling_compound(tokens, ner_tags, correct_with_quotes=False)
        print("results: {}".format(results))

        print()
        tokens = ['Click', 'on', 'BOQ', 'INSIDEQOUTES1', 'EOQ']
        ner_tags = ['O', 'O', 'O', 'O', 'O']
        results = ner_slot_filling_compound(tokens, ner_tags, correct_with_quotes=False)
        print("results: {}".format(results))

        print()
        tokens = ['Click', 'on', 'Commercial', 'driver', 'license', 'radio', 'moveandclick']
        ner_tags = ['B-ACT', 'O', 'B-OBJ', 'B-OBJ', 'B-OBJ', 'O', 'B-ACT']
        results = ner_slot_filling_compound(tokens, ner_tags, correct_with_quotes=False)
        print("results: {}".format(results))

if __name__ == '__main__':

    #unittest.main()

    test = Test_ner_slot_filling()
    #test.test_HOVER()
    #test.test_HOVER_missing_EOQ()
    #test.test_compound()
    test.test_compound_3()